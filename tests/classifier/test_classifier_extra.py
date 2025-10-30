import pytest
from unittest.mock import Mock, patch
from src.pipelines.classification.classifier import DocumentClassifier


class TestClassifierFailures:
    @pytest.fixture
    def clf(self):
        mock_q = Mock()
        return DocumentClassifier(mock_q, log_function=Mock()), mock_q

    def test_classify_no_collection_labels_found(self, clf):
        classifier, mock_q = clf
        with patch.object(classifier, '_load_labels_from_collection', return_value={}):
            result = classifier.classify_documents('c', use_collection_labels=True)
        assert result['success'] is False and 'No labels' in result['error']

    def test_classify_zero_vectors(self, clf):
        classifier, mock_q = clf
        # Labels file not used; supply labels via collection
        with patch.object(classifier, '_load_labels_from_collection', return_value={"1": {"label":"A"}}):
            mock_q.get_collection_info.return_value = {"vector_count": 0}
            result = classifier.classify_documents('c', use_collection_labels=True)
        assert result['success'] is False

    def test_classify_embed_failure(self, clf, tmp_path):
        classifier, mock_q = clf
        labels = {"1": {"label": "A"}}
        lp = tmp_path / 'labels.json'
        lp.write_text('{"1": {"label": "A"}}', encoding='utf-8')
        mock_q.get_collection_info.return_value = {"vector_count": 1}
        mock_q.scroll_vectors.return_value = ([{"id":1, "vector":[0.1], "payload":{}}], None)
        with patch('src.pipelines.classification.classifier.embed', return_value=None):
            result = classifier.classify_documents('c', labels_file=str(lp))
        assert result['success'] is False

    def test_classify_update_payload_failure(self, clf, tmp_path):
        classifier, mock_q = clf
        lp = tmp_path / 'labels.json'
        lp.write_text('{"1": {"label": "A"}}', encoding='utf-8')
        mock_q.get_collection_info.return_value = {"vector_count": 1}
        mock_q.scroll_vectors.return_value = ([{"id":1, "vector":[0.1], "payload":{}}], None)
        with patch('src.pipelines.classification.classifier.embed', return_value=[[0.1]]):
            mock_q.update_payload.return_value = False
            result = classifier.classify_documents('c', labels_file=str(lp))
        assert result['success'] is True and result['classified_count'] == 0

    def test_get_collection_labels_error(self, clf):
        classifier, mock_q = clf
        with patch.object(classifier, '_load_labels_from_collection', side_effect=Exception('boom')):
            result = classifier.get_collection_labels('c')
        assert result['success'] is False

    def test_add_label_without_enrich_empty_desc(self, clf):
        classifier, mock_q = clf
        mock_q._reserve_id_block.return_value = 10
        with patch('src.pipelines.classification.classifier.embed', return_value=[[0.1]]):
            with patch.object(classifier, '_generate_label_description') as gen:
                gen.return_value = 'GEN'
                res = classifier.add_label_to_collection('c', 'L', description=None, enrich=False)
        assert res['success'] is True

    def test_add_label_embed_failure(self, clf):
        classifier, mock_q = clf
        mock_q._reserve_id_block.return_value = 10
        with patch('src.pipelines.classification.classifier.embed', return_value=None):
            res = classifier.add_label_to_collection('c', 'L', description='D')
        assert res['success'] is False


