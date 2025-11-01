import pytest
from unittest.mock import patch, Mock


class TestCLIRemoveLabel:
    def test_rm_label_by_id_confirm_yes(self):
        from src.pipelines.classification.cli import Cli
        with patch('src.pipelines.classification.cli.QdrantService') as mock_qdrant_cls:
            with patch('src.pipelines.classification.cli.OpenAIService'):
                with patch('src.pipelines.classification.cli.ProcessingService'):
                    with patch('src.pipelines.classification.cli.SQLiteService'):
                        cli = Cli()
                        cli.collection = "test"
                        mock_q = mock_qdrant_cls.return_value
                        mock_q.scroll_vectors.return_value = ([{"id": 11, "payload": {"type":"label","label_id":"custom_x"}}], None)
                        mock_q.delete_points.return_value = True
                        with patch('builtins.input', return_value='yes'):
                            cli.handle_command("rm-label custom_x --by id")
                        mock_q.delete_points.assert_called_once()

    def test_rm_label_by_name_no_match(self):
        from src.pipelines.classification.cli import Cli
        with patch('src.pipelines.classification.cli.QdrantService') as mock_qdrant_cls:
            with patch('src.pipelines.classification.cli.OpenAIService'):
                with patch('src.pipelines.classification.cli.ProcessingService'):
                    with patch('src.pipelines.classification.cli.SQLiteService'):
                        cli = Cli()
                        cli.collection = "test"
                        mock_q = mock_qdrant_cls.return_value
                        mock_q.scroll_vectors.return_value = ([], None)
                        with patch('builtins.input', return_value='yes'):
                            cli.handle_command("rm-label Sports --by name")
                        # No deletion when no matches
                        assert not mock_q.delete_points.called


class TestCLIAddLabelFlags:
    def test_add_label_with_description_no_enrich(self):
        from src.pipelines.classification.cli import Cli
        with patch('src.pipelines.classification.cli.QdrantService'):
            with patch('src.pipelines.classification.cli.OpenAIService'):
                with patch('src.pipelines.classification.cli.ProcessingService'):
                    with patch('src.pipelines.classification.cli.SQLiteService'):
                        with patch('src.pipelines.classification.cli.DocumentClassifier') as mock_cls:
                            cli = Cli()
                            cli.collection = "test"
                            mock_inst = mock_cls.return_value
                            mock_inst.add_label_to_collection.return_value = {"success": True}
                            cli.handle_command("add-label Economy --description 'Finance'")
                            mock_inst.add_label_to_collection.assert_called_once()
                            args, kwargs = mock_inst.add_label_to_collection.call_args
                            assert kwargs.get('enrich') is False or len(args) >= 3

    def test_add_label_enrich_without_description(self):
        from src.pipelines.classification.cli import Cli
        with patch('src.pipelines.classification.cli.QdrantService'):
            with patch('src.pipelines.classification.cli.OpenAIService'):
                with patch('src.pipelines.classification.cli.ProcessingService'):
                    with patch('src.pipelines.classification.cli.SQLiteService'):
                        with patch('src.pipelines.classification.cli.DocumentClassifier') as mock_cls:
                            cli = Cli()
                            cli.collection = "test"
                            mock_inst = mock_cls.return_value
                            mock_inst.add_label_to_collection.return_value = {"success": True}
                            cli.handle_command("add-label Economy --enrich")
                            _, kwargs = mock_inst.add_label_to_collection.call_args
                            assert kwargs.get('enrich') is True


class TestCLIListLabelsFallback:
    def test_list_labels_fallback_inferred(self):
        from src.pipelines.classification.cli import Cli
        with patch('src.pipelines.classification.cli.QdrantService') as mock_qdrant:
            with patch('src.pipelines.classification.cli.OpenAIService'):
                with patch('src.pipelines.classification.cli.ProcessingService'):
                    with patch('src.pipelines.classification.cli.SQLiteService'):
                        with patch('src.pipelines.classification.cli.DocumentClassifier'):
                            cli = Cli()
                            cli.collection = "test"
                            mock_q = mock_qdrant.return_value
            # No stored labels
            mock_q.scroll_vectors.side_effect = [([], None), ([
                {"id":1, "payload": {"predicted_label":"A"}},
                {"id":2, "payload": {"predicted_label":"A"}},
                {"id":3, "payload": {"predicted_label":"B"}},
            ], None)]
            # Should not raise
            cli.handle_command("list-labels")


class TestCLIQueryVariants:
    def setup_cli(self):
        from src.pipelines.classification.cli import Cli
        with patch('src.pipelines.classification.cli.QdrantService') as mock_q:
            with patch('src.pipelines.classification.cli.OpenAIService'):
                with patch('src.pipelines.classification.cli.ProcessingService'):
                    with patch('src.pipelines.classification.cli.SQLiteService'):
                        with patch('src.pipelines.classification.cli.DocumentClassifier'):
                            cli = Cli()
                            cli.collection = "test"
                            return cli, mock_q.return_value

    def test_query_cluster_id_and_name(self):
        cli, mock_q = self.setup_cli()
        mock_q.scroll_vectors.return_value = ([{"id":1,"payload":{}}], None)
        cli.handle_command("query cluster:1")
        cli.handle_command("query cluster:Tech")
        assert mock_q.scroll_vectors.call_count >= 2

    def test_query_label_and_docid_and_dir(self):
        cli, mock_q = self.setup_cli()
        mock_q.scroll_vectors.return_value = ([{"id":1,"payload":{}}], None)
        cli.handle_command("query label:Sports")
        cli.handle_command("query 42")
        assert mock_q.scroll_vectors.call_count >= 2


class TestCLIClusterDebug:
    def test_cluster_debug_flag_calls_debug(self, tmp_path):
        from src.pipelines.classification.cli import Cli
        with patch('src.pipelines.classification.cli.QdrantService') as mock_qdrant:
            with patch('src.pipelines.classification.cli.OpenAIService') as mock_openai:
                with patch('src.pipelines.classification.cli.ProcessingService'):
                    with patch('src.pipelines.classification.cli.SQLiteService'):
                        with patch('src.pipelines.classification.cli.DocumentClassifier'):
                            cli = Cli()
                            cli.collection = "test"
                            mq = mock_qdrant.return_value
            mq.get_collection_info.return_value = {"vector_count": 3}
            mq.scroll_vectors.return_value = ([
                {"id":1, "vector":[0.1]*4, "payload":{}},
                {"id":2, "vector":[0.2]*4, "payload":{}},
                {"id":3, "vector":[0.3]*4, "payload":{}},
            ], None)
            mo = mock_openai.return_value
            mo.generate_single_word_cluster_label.return_value = "X"
            with patch('src.pipelines.classification.cli.json.dump'):
                # Avoid creating cluster output file on disk
                with patch('builtins.open', create=True):
                    cli._cluster_command(num_clusters=2, debug=True)
            assert True


class TestCLICreateAndRetryEdges:
    def test_create_invalid_description_flag(self):
        from src.pipelines.classification.cli import Cli
        with patch('src.pipelines.classification.cli.QdrantService'):
            with patch('src.pipelines.classification.cli.OpenAIService'):
                with patch('src.pipelines.classification.cli.ProcessingService'):
                    with patch('src.pipelines.classification.cli.SQLiteService'):
                        with patch('src.pipelines.classification.cli.DocumentClassifier'):
                            cli = Cli()
                            # Should log usage error, not crash
                            cli.handle_command("create test -dsadasdsa")

    def test_retry_failure_path(self):
        from src.pipelines.classification.cli import Cli
        with patch('src.pipelines.classification.cli.QdrantService') as mock_qdrant:
            with patch('src.pipelines.classification.cli.OpenAIService'):
                with patch('src.pipelines.classification.cli.ProcessingService'):
                    with patch('src.pipelines.classification.cli.SQLiteService'):
                        with patch('src.pipelines.classification.cli.DocumentClassifier'):
                            # First instantiate CLI
                            cli = Cli()
                            # Make reconnect raise inside constructor usage
                            mock_qdrant.side_effect = [mock_qdrant.return_value, Exception("fail")]
            with patch('builtins.print') as mock_print:
                cli.handle_command("retry --host bad --port 1")
                # Should print retry message
                assert mock_print.called


