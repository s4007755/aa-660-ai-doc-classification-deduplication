from src.services.qdrant_service import QdrantService
svc = QdrantService("localhost",6333,print)
coll='test5'
pts, _ = svc.scroll_vectors(coll, 1000, with_payload=True, with_vectors=False, filter_conditions={"type":"label"})
print('labeled points:', len(pts))
pts_all, _ = svc.scroll_vectors(coll, 20000, with_payload=True, with_vectors=False)
print('all points:', len(pts_all))
labels_by_key = [p for p in pts_all if isinstance(p.get('payload'), dict) and ('label_name' in p['payload'] or p['payload'].get('type')=='label')]
print('heuristic label-like:', len(labels_by_key))
if labels_by_key:
    print('example payload:', {k:labels_by_key[0]['payload'].get(k) for k in ('type','label_id','label_name','description')})
