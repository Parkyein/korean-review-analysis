import json
from collections import defaultdict

# ──────────────────────────────────────────
# 설정
# ──────────────────────────────────────────

INPUT_PATH  = "/Users/parkyein/Desktop/소셜_기말프로젝트/enriched_all_final.json"
OUTPUT_PATH = "/Users/parkyein/Desktop/소셜_기말프로젝트/V3_enriched_graph.json"

# contrast direction 판단용 양보 키워드 (엄격하게)
POSITIVE_DOMINANT_KEYWORDS = [
    "그래도", "그나마", "참을만", "아쉽지만", "아쉽긴 하지만",
    "그럼에도 불구하고", "어쩔 수 없이"
]

# expectation 트리거 키워드 (방향 판단은 aspect_sentiment로)
EXPECTATION_KEYWORDS = [
    "기대했는데", "기대했지만", "기대 이하", "기대 이상",
    "라더니", "줄 알았는데", "생각보다", "예상보다",
    "광고랑 다르", "사진이랑 다", "의외로"
]

# ──────────────────────────────────────────
# Layer 1 빌더
# ──────────────────────────────────────────

def get_direction(content: str) -> str:
    for kw in POSITIVE_DOMINANT_KEYWORDS:
        if kw in content:
            return "positive_dominant"
    return "negative_dominant"


def build_layer1_edges(record: dict) -> list:
    edges = []
    content = record.get("content", "")
    enriched = record.get("enriched", {})
    aspect_relations = enriched.get("aspect_relations", []) or []
    mismatch = enriched.get("rating_content_mismatch", {}) or {}
    rating = record.get("rating", 3)
    asp_sents = enriched.get("aspect_sentiments", {}) or {}

    # ① cause / ② contrast (기존 aspect_relations 변환)
    for rel in aspect_relations:
        rel_type = rel.get("type")
        left     = rel.get("left", "")
        right    = rel.get("right", "")
        evidence = rel.get("evidence", "")
        conf     = rel.get("confidence", 0.8)

        if rel_type == "cause":
            external_aspects = {"delivery", "packaging", "service", "benefit_issue", "product_condition"}
            is_mismatch = (
                mismatch.get("label") == True
                and any(ext in left for ext in external_aspects)
                and "low_rating" in right
            )
            if is_mismatch:
                edges.append({
                    "edge_type": "mismatch_cause",
                    "source": left,
                    "target": right,
                    "mismatch_confirmed": True,
                    "evidence": evidence,
                    "confidence": conf,
                    "method": "llm+rule"
                })
            else:
                edges.append({
                    "edge_type": "cause",
                    "source": left,
                    "target": right,
                    "evidence": evidence,
                    "confidence": conf,
                    "method": "llm"
                })

        elif rel_type == "contrast":
            direction = get_direction(content)
            matched_kw = [kw for kw in POSITIVE_DOMINANT_KEYWORDS if kw in content]
            edges.append({
                "edge_type": "contrast",
                "source": left,
                "target": right,
                "direction": direction,
                "keywords_matched": matched_kw if matched_kw else None,
                "evidence": evidence,
                "confidence": conf,
                "method": "llm+rule"
            })

    # rule-based cause 보완 (rating<=2 + 외부 부정 aspect, 기존 cause 없으면)
    has_cause = any(e["edge_type"] in ("cause", "mismatch_cause") for e in edges)
    if not has_cause and rating <= 2:
        external_aspects = ["delivery", "packaging", "service", "product_condition"]
        for asp in external_aspects:
            if asp_sents.get(asp, {}).get("sentiment") == "negative":
                edges.append({
                    "edge_type": "cause",
                    "source": f"{asp}_negative",
                    "target": "low_rating",
                    "evidence": asp_sents[asp].get("evidence", ""),
                    "confidence": 0.7,
                    "method": "rule_based"
                })
                break

    # ③ expectation_fail / ④ expectation_meet
    exp_kws = [kw for kw in EXPECTATION_KEYWORDS if kw in content]
    if exp_kws:
        for asp, v in asp_sents.items():
            evidence = v.get("evidence", "")
            if not any(kw in evidence for kw in exp_kws):  # ← 여기만 바뀜
                continue
            if v.get("sentiment") == "negative":
                edges.append({
                    "edge_type": "expectation_fail",
                    "source": f"{asp}_high_expectation",
                    "target": f"{asp}_negative",
                    "keywords_matched": exp_kws,
                    "confidence": 0.8,
                    "method": "rule_based"
                })
            elif v.get("sentiment") == "positive":
                edges.append({
                    "edge_type": "expectation_meet",
                    "source": f"{asp}_high_expectation",
                    "target": f"{asp}_positive",
                    "keywords_matched": exp_kws,
                    "confidence": 0.8,
                    "method": "rule_based"
                })

    return edges


def build_layer1_nodes(record: dict, edges: list) -> list:
    enriched  = record.get("enriched", {})
    asp_sents = enriched.get("aspect_sentiments", {}) or {}
    node_names = set()
    for e in edges:
        node_names.add(e["source"])
        node_names.add(e["target"])

    nodes = []
    for name in node_names:
        if "high_expectation" in name:
            asp = name.replace("_high_expectation", "")
            nodes.append({
                "node_name": name,
                "node_type": "high_expectation",
                "target": asp
            })
        elif name in ("low_rating", "high_rating"):
            nodes.append({
                "node_name": name,
                "node_type": "rating_signal",
                "polarity": "negative" if "low" in name else "positive"
            })
        else:
            for pol in ("positive", "negative", "neutral"):
                if name.endswith(f"_{pol}"):
                    asp = name[: -(len(pol) + 1)]
                    conf = asp_sents.get(asp, {}).get("confidence", None)
                    nodes.append({
                        "node_name": name,
                        "node_type": "aspect_sentiment",
                        "target": asp,
                        "polarity": pol,
                        "confidence": conf
                    })
                    break
    return nodes


# ──────────────────────────────────────────
# Layer 2 빌더
# ──────────────────────────────────────────

def build_layer2_mentions(record: dict) -> list:
    enriched  = record.get("enriched", {})
    asp_sents = enriched.get("aspect_sentiments", {}) or {}
    mismatch  = enriched.get("rating_content_mismatch", {}) or {}
    edges = []
    for asp, info in asp_sents.items():
        pol = info.get("sentiment", "neutral")
        edges.append({
            "edge_type": "mentions",
            "source": record["review_id"],
            "target": f"{asp}_{pol}",
            "rating": record.get("rating"),
            "product_name": record.get("product_name"),
            "has_mismatch": mismatch.get("label", False)
        })
    return edges


def build_layer2_switched_to(records: list) -> list:
    agg = defaultdict(lambda: {
        "count": 0, "sentiments": [], "ratings": [], "aspects": defaultdict(int)
    })

    for record in records:
        mig = record.get("enriched", {}).get("migration_info")
        if not mig:
            continue
        frm  = mig.get("from")
        to   = mig.get("to")
        sent = mig.get("sentiment", "neutral")
        if not frm or not to:
            continue

        key = (frm, to)
        agg[key]["count"] += 1
        agg[key]["sentiments"].append(sent)
        agg[key]["ratings"].append(record.get("rating", 3))

        asp_sents = record.get("enriched", {}).get("aspect_sentiments", {}) or {}
        for asp, info in asp_sents.items():
            if info.get("sentiment") == "positive":
                agg[key]["aspects"][f"{asp}_positive"] += 1

    edges = []
    for (frm, to), v in agg.items():
        sents = v["sentiments"]
        sent_dist = {
            "positive": sents.count("positive"),
            "negative": sents.count("negative"),
            "neutral":  sents.count("neutral")
        }
        top_aspects = sorted(v["aspects"], key=v["aspects"].get, reverse=True)[:3]
        avg_rating  = round(sum(v["ratings"]) / len(v["ratings"]), 2) if v["ratings"] else None

        edges.append({
            "edge_type": "switched_to",
            "source": frm,
            "target": to,
            "count": v["count"],
            "sentiment_dist": sent_dist,
            "top_aspects": top_aspects,
            "avg_rating": avg_rating
        })

    return edges


def build_cooccurrence(records_with_graph: list, threshold: int = 10) -> list:
    co_count = defaultdict(int)
    for rec in records_with_graph:
        mentions = [
            e["target"] for e in rec["experience_graph"]["layer2_edges"]
            if e["edge_type"] == "mentions"
        ]
        for i in range(len(mentions)):
            for j in range(i + 1, len(mentions)):
                pair = tuple(sorted([mentions[i], mentions[j]]))
                co_count[pair] += 1

    edges = []
    for (a, b), cnt in co_count.items():
        if cnt >= threshold:
            edges.append({
                "edge_type": "co_occurs_with",
                "source": a,
                "target": b,
                "weight": cnt
            })
    return sorted(edges, key=lambda x: -x["weight"])


# ──────────────────────────────────────────
# 메인
# ──────────────────────────────────────────

def main():
    with open(INPUT_PATH, encoding="utf-8") as f:
        data = json.load(f)

    print(f"총 {len(data)}건 처리 시작...")

    switched_to_edges = build_layer2_switched_to(data)

    results = []
    for record in data:
        l1_edges    = build_layer1_edges(record)
        l1_nodes    = build_layer1_nodes(record, l1_edges)
        l2_mentions = build_layer2_mentions(record)

        record["experience_graph"] = {
            "layer1_nodes": l1_nodes,
            "layer1_edges": l1_edges,
            "layer2_edges": l2_mentions
        }
        results.append(record)

    co_edges = build_cooccurrence(results, threshold=10)
    print(f"co_occurs_with 엣지: {len(co_edges)}개")

    output = {
        "records": results,
        "global_graph": {
            "switched_to_edges": switched_to_edges,
            "co_occurs_with_edges": co_edges
        }
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    total_l1 = sum(len(r["experience_graph"]["layer1_edges"]) for r in results)
    total_l2 = sum(len(r["experience_graph"]["layer2_edges"]) for r in results)
    edge_type_count = defaultdict(int)
    for r in results:
        for e in r["experience_graph"]["layer1_edges"]:
            edge_type_count[e["edge_type"]] += 1

    print(f"\n=== 생성 완료 ===")
    print(f"Layer 1 엣지 합계: {total_l1}")
    print(f"Layer 2 mentions:  {total_l2}")
    print(f"switched_to:       {len(switched_to_edges)}")
    print(f"co_occurs_with:    {len(co_edges)}")
    print("\n[ Layer 1 엣지 타입별 ]")
    for et, cnt in sorted(edge_type_count.items(), key=lambda x: -x[1]):
        print(f"  {et:25s}: {cnt}")
    print(f"\n출력 파일: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()