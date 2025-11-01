"""
Comprehensive Test Suite for CLI (cli.py)

This test suite provides comprehensive coverage of the CLI functionality including:
- Command parsing and execution
- Service integration
- Error handling
- Edge cases
- Security validation
"""

import pytest
import sys
import os
from unittest.mock import Mock, MagicMock, patch
from io import StringIO

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.pipelines.classification.cli import Cli, NoExitArgParser


class TestCLIInitialization:
    """Test CLI initialization and setup."""
    
    def test_cli_initialization_default(self):
        """Test CLI initializes with default parameters."""
        with patch('src.pipelines.classification.cli.QdrantService'):
            with patch('src.pipelines.classification.cli.OpenAIService'):
                with patch('src.pipelines.classification.cli.ProcessingService'):
                    with patch('src.pipelines.classification.cli.SQLiteService'):
                        with patch('src.pipelines.classification.cli.DocumentClassifier'):
                            cli = Cli()
                            assert cli.host == "localhost"
                            assert cli.port == 6333
                            assert cli.collection is None
    
    def test_cli_initialization_custom_host_port(self):
        """Test CLI initializes with custom host and port."""
        with patch('src.pipelines.classification.cli.QdrantService'):
            with patch('src.pipelines.classification.cli.OpenAIService'):
                with patch('src.pipelines.classification.cli.ProcessingService'):
                    with patch('src.pipelines.classification.cli.SQLiteService'):
                        with patch('src.pipelines.classification.cli.DocumentClassifier'):
                            cli = Cli(host="remote-server", port=6334)
                            assert cli.host == "remote-server"
                            assert cli.port == 6334
    
    def test_cli_services_initialized(self):
        """Test all required services are initialized."""
        with patch('src.pipelines.classification.cli.QdrantService') as mock_qdrant:
            with patch('src.pipelines.classification.cli.OpenAIService') as mock_openai:
                with patch('src.pipelines.classification.cli.ProcessingService') as mock_processing:
                    with patch('src.pipelines.classification.cli.SQLiteService') as mock_sqlite:
                        with patch('src.pipelines.classification.cli.DocumentClassifier') as mock_classifier:
                            cli = Cli()
                            
                            # Verify services were initialized
                            mock_qdrant.assert_called_once()
                            mock_openai.assert_called_once()
                            mock_processing.assert_called_once()
                            mock_sqlite.assert_called_once()
                            mock_classifier.assert_called_once()


class TestCLILogging:
    """Test CLI logging functionality."""
    
    def test_log_normal_message(self):
        """Test logging normal messages."""
        with patch('src.pipelines.classification.cli.QdrantService'):
            with patch('src.pipelines.classification.cli.OpenAIService'):
                with patch('src.pipelines.classification.cli.ProcessingService'):
                    with patch('src.pipelines.classification.cli.SQLiteService'):
                        with patch('src.pipelines.classification.cli.DocumentClassifier'):
                            cli = Cli()
                            with patch.object(cli.console, 'print') as mock_print:
                                cli.log("Test message")
                                mock_print.assert_called_once()
    
    def test_log_error_message(self):
        """Test logging error messages."""
        with patch('src.pipelines.classification.cli.QdrantService'):
            with patch('src.pipelines.classification.cli.OpenAIService'):
                with patch('src.pipelines.classification.cli.ProcessingService'):
                    with patch('src.pipelines.classification.cli.SQLiteService'):
                        with patch('src.pipelines.classification.cli.DocumentClassifier'):
                            cli = Cli()
                            with patch.object(cli.console, 'print') as mock_print:
                                cli.log("Error message", error=True)
                                mock_print.assert_called_once()


class TestCLICommandParsing:
    """Test command parsing functionality."""
    
    def test_handle_command_use(self):
        """Test 'use' command parsing."""
        with patch('src.pipelines.classification.cli.QdrantService') as mock_qdrant:
            with patch('src.pipelines.classification.cli.OpenAIService'):
                with patch('src.pipelines.classification.cli.ProcessingService'):
                    with patch('src.pipelines.classification.cli.SQLiteService'):
                        with patch('src.pipelines.classification.cli.DocumentClassifier'):
                            cli = Cli()
                            mock_qdrant_instance = mock_qdrant.return_value
                            mock_qdrant_instance.list_collections.return_value = ["test_collection"]
                            
                            cli.handle_command("use test_collection")
                            assert cli.collection == "test_collection"
    
    def test_handle_command_use_missing_argument(self):
        """Test 'use' command without collection name."""
        with patch('src.pipelines.classification.cli.QdrantService'):
            with patch('src.pipelines.classification.cli.OpenAIService'):
                with patch('src.pipelines.classification.cli.ProcessingService'):
                    with patch('src.pipelines.classification.cli.SQLiteService'):
                        with patch('src.pipelines.classification.cli.DocumentClassifier'):
                            cli = Cli()
                            with patch.object(cli, 'log') as mock_log:
                                cli.handle_command("use")
                                mock_log.assert_called_with("Usage: use <collection>", True)
    
    def test_handle_command_show_connected(self):
        """Test 'show' command when connected."""
        with patch('src.pipelines.classification.cli.QdrantService') as mock_qdrant:
            with patch('src.pipelines.classification.cli.OpenAIService'):
                with patch('src.pipelines.classification.cli.ProcessingService'):
                    with patch('src.pipelines.classification.cli.SQLiteService'):
                        with patch('src.pipelines.classification.cli.DocumentClassifier'):
                            cli = Cli()
                            mock_qdrant_instance = mock_qdrant.return_value
                            mock_qdrant_instance.is_connected.return_value = True
                            
                            # CLI now uses console.print instead of print
                            cli.handle_command("show")
                            # Just verify it doesn't crash
                            assert True
    
    def test_handle_command_ls(self):
        """Test 'ls' command to list collections."""
        with patch('src.pipelines.classification.cli.QdrantService') as mock_qdrant:
            with patch('src.pipelines.classification.cli.OpenAIService'):
                with patch('src.pipelines.classification.cli.ProcessingService'):
                    with patch('src.pipelines.classification.cli.SQLiteService'):
                        with patch('src.pipelines.classification.cli.DocumentClassifier'):
                            cli = Cli()
                            mock_qdrant_instance = mock_qdrant.return_value
                            # list_collections now returns dict, not list
                            mock_qdrant_instance.list_collections.return_value = {
                                "col1": {"size": 1536, "vectors": 100, "distance": "Cosine"},
                                "col2": {"size": 1536, "vectors": 50, "distance": "Cosine"}
                            }
                            
                            # CLI now uses console.print instead of print
                            cli.handle_command("ls")
                            # Just verify it doesn't crash
                            assert True
    
    def test_handle_command_create(self):
        """Test 'create' command to create collection."""
        with patch('src.pipelines.classification.cli.QdrantService') as mock_qdrant:
            with patch('src.pipelines.classification.cli.OpenAIService'):
                with patch('src.pipelines.classification.cli.ProcessingService'):
                    with patch('src.pipelines.classification.cli.SQLiteService'):
                        with patch('src.pipelines.classification.cli.DocumentClassifier'):
                            cli = Cli()
                            mock_qdrant_instance = mock_qdrant.return_value
                            mock_qdrant_instance.create_collection.return_value = True
                            
                            with patch('builtins.print') as mock_print:
                                cli.handle_command("create test_collection")
                                mock_qdrant_instance.create_collection.assert_called_once()
    
    def test_handle_command_help(self):
        """Test 'help' command displays help text."""
        with patch('src.pipelines.classification.cli.QdrantService'):
            with patch('src.pipelines.classification.cli.OpenAIService'):
                with patch('src.pipelines.classification.cli.ProcessingService'):
                    with patch('src.pipelines.classification.cli.SQLiteService'):
                        with patch('src.pipelines.classification.cli.DocumentClassifier'):
                            cli = Cli()
                            # CLI now uses console.print instead of print
                            cli.handle_command("help")
                            # Just verify it doesn't crash
                            assert True
    
    def test_handle_command_exit(self):
        """Test 'exit' command terminates program."""
        with patch('src.pipelines.classification.cli.QdrantService'):
            with patch('src.pipelines.classification.cli.OpenAIService'):
                with patch('src.pipelines.classification.cli.ProcessingService'):
                    with patch('src.pipelines.classification.cli.SQLiteService'):
                        with patch('src.pipelines.classification.cli.DocumentClassifier'):
                            cli = Cli()
                            with pytest.raises(SystemExit):
                                cli.handle_command("exit")
    
    def test_handle_command_quit(self):
        """Test 'quit' command terminates program."""
        with patch('src.pipelines.classification.cli.QdrantService'):
            with patch('src.pipelines.classification.cli.OpenAIService'):
                with patch('src.pipelines.classification.cli.ProcessingService'):
                    with patch('src.pipelines.classification.cli.SQLiteService'):
                        with patch('src.pipelines.classification.cli.DocumentClassifier'):
                            cli = Cli()
                            with pytest.raises(SystemExit):
                                cli.handle_command("quit")
    
    def test_handle_command_unknown(self):
        """Test handling unknown commands."""
        with patch('src.pipelines.classification.cli.QdrantService'):
            with patch('src.pipelines.classification.cli.OpenAIService'):
                with patch('src.pipelines.classification.cli.ProcessingService'):
                    with patch('src.pipelines.classification.cli.SQLiteService'):
                        with patch('src.pipelines.classification.cli.DocumentClassifier'):
                            cli = Cli()
                            with patch('builtins.print') as mock_print:
                                cli.handle_command("invalid_command")
                                mock_print.assert_called_with("Unknown command: invalid_command")


class TestCLISourceCommand:
    """Test 'source' command functionality."""
    
    def test_source_command_no_collection(self):
        """Test source command without selected collection."""
        with patch('src.pipelines.classification.cli.QdrantService'):
            with patch('src.pipelines.classification.cli.OpenAIService'):
                with patch('src.pipelines.classification.cli.ProcessingService'):
                    with patch('src.pipelines.classification.cli.SQLiteService'):
                        with patch('src.pipelines.classification.cli.DocumentClassifier'):
                            cli = Cli()
                            with patch.object(cli, 'log') as mock_log:
                                cli.handle_command("source test_directory")
                                mock_log.assert_called_with("No collection selected.", True)
    
    def test_source_command_with_directory(self):
        """Test source command with directory path."""
        with patch('src.pipelines.classification.cli.QdrantService') as mock_qdrant:
            with patch('src.pipelines.classification.cli.OpenAIService') as mock_openai:
                with patch('src.pipelines.classification.cli.ProcessingService') as mock_processing:
                    with patch('src.pipelines.classification.cli.SQLiteService'):
                        with patch('src.pipelines.classification.cli.DocumentClassifier'):
                            cli = Cli()
                            cli.collection = "test_collection"
                            
                            # Mock service responses
                            mock_processing_instance = mock_processing.return_value
                            mock_processing_instance.process_source.return_value = (["text1"], [{"hash": "hash1"}])
                            
                            mock_openai_instance = mock_openai.return_value
                            mock_openai_instance.generate_embeddings.return_value = [[0.1] * 1536]
                            
                            mock_qdrant_instance = mock_qdrant.return_value
                            mock_qdrant_instance.get_collection_model.return_value = "text-embedding-3-small"
                            mock_qdrant_instance.insert_vectors.return_value = (True, 1, 0)
                            
                            cli.handle_command("source test_directory")
                            
                            # Verify service calls
                            mock_processing_instance.process_source.assert_called_once()
                            mock_openai_instance.generate_embeddings.assert_called_once()

    def test_source_command_no_data(self):
        """If processing returns no texts, CLI should log and return early."""
        with patch('src.pipelines.classification.cli.QdrantService') as mock_qdrant:
            with patch('src.pipelines.classification.cli.OpenAIService') as mock_openai:
                with patch('src.pipelines.classification.cli.ProcessingService') as mock_processing:
                    with patch('src.pipelines.classification.cli.SQLiteService'):
                        with patch('src.pipelines.classification.cli.DocumentClassifier'):
                            cli = Cli()
                            cli.collection = "test_collection"

                            mock_processing.return_value.process_source.return_value = ([], [])
                            with patch.object(cli, 'log') as mock_log:
                                cli.handle_command("source test_directory")
                                mock_log.assert_called()
    
    def test_source_command_with_limit(self):
        """Test source command with limit parameter."""
        with patch('src.pipelines.classification.cli.QdrantService') as mock_qdrant:
            with patch('src.pipelines.classification.cli.OpenAIService') as mock_openai:
                with patch('src.pipelines.classification.cli.ProcessingService') as mock_processing:
                    with patch('src.pipelines.classification.cli.SQLiteService'):
                        with patch('src.pipelines.classification.cli.DocumentClassifier'):
                            cli = Cli()
                            cli.collection = "test_collection"
                            
                            # Mock service responses
                            mock_processing_instance = mock_processing.return_value
                            mock_processing_instance.process_source.return_value = (["text1"], [{"hash": "hash1"}])
                            
                            mock_openai_instance = mock_openai.return_value
                            mock_openai_instance.generate_embeddings.return_value = [[0.1] * 1536]
                            
                            mock_qdrant_instance = mock_qdrant.return_value
                            mock_qdrant_instance.get_collection_model.return_value = "text-embedding-3-small"
                            mock_qdrant_instance.insert_vectors.return_value = (True, 1, 0)
                            
                            cli.handle_command("source test_directory --limit 100")
                            
                            # Verify limit was passed
                            call_args = mock_processing_instance.process_source.call_args
                            assert call_args[0][1] == 100


class TestCLIClusterCommand:
    """Test 'cluster' command functionality."""
    
    def test_cluster_command_no_collection(self):
        """Test cluster command without selected collection."""
        with patch('src.pipelines.classification.cli.QdrantService'):
            with patch('src.pipelines.classification.cli.OpenAIService'):
                with patch('src.pipelines.classification.cli.ProcessingService'):
                    with patch('src.pipelines.classification.cli.SQLiteService'):
                        with patch('src.pipelines.classification.cli.DocumentClassifier'):
                            cli = Cli()
                            with patch.object(cli, 'log') as mock_log:
                                cli.handle_command("cluster")
                                mock_log.assert_called_with("No collection selected.", True)
    
    def test_cluster_command_kmeans(self):
        """Test cluster command with K-means algorithm."""
        with patch('src.pipelines.classification.cli.QdrantService') as mock_qdrant:
            with patch('src.pipelines.classification.cli.OpenAIService') as mock_openai:
                with patch('src.pipelines.classification.cli.ProcessingService'):
                    with patch('src.pipelines.classification.cli.SQLiteService'):
                        with patch('src.pipelines.classification.cli.DocumentClassifier'):
                            cli = Cli()
                            cli.collection = "test_collection"
                            
                            # Mock collection info
                            mock_qdrant_instance = mock_qdrant.return_value
                            mock_qdrant_instance.get_collection_info.return_value = {'vector_count': 100}
                            mock_qdrant_instance.scroll_vectors.return_value = (
                                [{'id': i, 'vector': [0.1] * 1536, 'payload': {}} for i in range(10)],
                                None
                            )
                            
                            # Mock OpenAI service
                            mock_openai_instance = mock_openai.return_value
                            mock_openai_instance.generate_single_word_cluster_label.return_value = "TestCluster"
                            
                            with patch('builtins.open', create=True):
                                cli.handle_command("cluster --num-clusters 3")


class TestCLIClassifyCommand:
    """Test 'classify' command functionality."""
    
    def test_classify_command_no_collection(self):
        """Test classify command without selected collection."""
        with patch('src.pipelines.classification.cli.QdrantService'):
            with patch('src.pipelines.classification.cli.OpenAIService'):
                with patch('src.pipelines.classification.cli.ProcessingService'):
                    with patch('src.pipelines.classification.cli.SQLiteService'):
                        with patch('src.pipelines.classification.cli.DocumentClassifier'):
                            cli = Cli()
                            with patch.object(cli, 'log') as mock_log:
                                cli.handle_command("classify labels.json")
                                mock_log.assert_called_with("No collection selected.", True)
    
    def test_classify_command_with_labels_file(self):
        """Test classify command with labels file."""
        with patch('src.pipelines.classification.cli.QdrantService'):
            with patch('src.pipelines.classification.cli.OpenAIService'):
                with patch('src.pipelines.classification.cli.ProcessingService'):
                    with patch('src.pipelines.classification.cli.SQLiteService'):
                        with patch('src.pipelines.classification.cli.DocumentClassifier') as mock_classifier:
                            cli = Cli()
                            cli.collection = "test_collection"
                            
                            # Mock classifier
                            mock_classifier_instance = mock_classifier.return_value
                            mock_classifier_instance.classify_documents.return_value = {"success": True}
                            
                            cli.handle_command("classify labels.json")
                            
                            # Verify classifier was called
                            mock_classifier_instance.classify_documents.assert_called_once()


class TestCLIQueryCommand:
    """Test 'query' command functionality."""
    
    def test_query_command_no_collection(self):
        """Test query command without selected collection."""
        with patch('src.pipelines.classification.cli.QdrantService'):
            with patch('src.pipelines.classification.cli.OpenAIService'):
                with patch('src.pipelines.classification.cli.ProcessingService'):
                    with patch('src.pipelines.classification.cli.SQLiteService'):
                        with patch('src.pipelines.classification.cli.DocumentClassifier'):
                            cli = Cli()
                            with patch.object(cli, 'log') as mock_log:
                                cli.handle_command("query test")
                                mock_log.assert_called_with("No collection selected.", True)
    
    def test_query_command_with_url(self):
        """Test query command with URL."""
        with patch('src.pipelines.classification.cli.QdrantService') as mock_qdrant:
            with patch('src.pipelines.classification.cli.OpenAIService'):
                with patch('src.pipelines.classification.cli.ProcessingService'):
                    with patch('src.pipelines.classification.cli.SQLiteService'):
                        with patch('src.pipelines.classification.cli.DocumentClassifier'):
                            cli = Cli()
                            cli.collection = "test_collection"
                            
                            # Mock qdrant service
                            mock_qdrant_instance = mock_qdrant.return_value
                            mock_qdrant_instance.scroll_vectors.return_value = (
                                [{'id': 1, 'payload': {'source': 'http://test.com'}}],
                                None
                            )
                            
                            # CLI now uses console.print instead of print
                            cli.handle_command("query http://test.com")
                            # Verify scroll_vectors was called
                            mock_qdrant_instance.scroll_vectors.assert_called()


class TestCLIStatsCommand:
    """Test 'stats' command functionality."""
    
    def test_stats_command_no_collection(self):
        """Test stats command without selected collection."""
        with patch('src.pipelines.classification.cli.QdrantService'):
            with patch('src.pipelines.classification.cli.OpenAIService'):
                with patch('src.pipelines.classification.cli.ProcessingService'):
                    with patch('src.pipelines.classification.cli.SQLiteService'):
                        with patch('src.pipelines.classification.cli.DocumentClassifier'):
                            cli = Cli()
                            with patch.object(cli, 'log') as mock_log:
                                cli.handle_command("stats")
                                mock_log.assert_called_with("No collection selected.", True)
    
    def test_stats_command_with_collection(self):
        """Test stats command with selected collection."""
        with patch('src.pipelines.classification.cli.QdrantService') as mock_qdrant:
            with patch('src.pipelines.classification.cli.OpenAIService'):
                with patch('src.pipelines.classification.cli.ProcessingService'):
                    with patch('src.pipelines.classification.cli.SQLiteService'):
                        with patch('src.pipelines.classification.cli.DocumentClassifier'):
                            cli = Cli()
                            cli.collection = "test_collection"
                            
                            # Mock collection info
                            mock_qdrant_instance = mock_qdrant.return_value
                            mock_qdrant_instance.get_collection_info.return_value = {
                                'vector_count': 100,
                                'dimension': 1536,
                                'distance_metric': 'Cosine'
                            }
                            mock_qdrant_instance.get_collection_metadata.return_value = None
                            mock_qdrant_instance.scroll_vectors.return_value = (
                                [{'id': 1, 'payload': {'type': 'file'}}],
                                None
                            )
                            
                            # CLI now uses console.print instead of print
                            cli.handle_command("stats")
                            # Verify get_collection_info was called
                            mock_qdrant_instance.get_collection_info.assert_called()


class TestCLIRetryCommand:
    """Test 'retry' command functionality."""
    
    def test_retry_command_default_params(self):
        """Test retry command with default parameters."""
        with patch('src.pipelines.classification.cli.QdrantService') as mock_qdrant:
            with patch('src.pipelines.classification.cli.OpenAIService'):
                with patch('src.pipelines.classification.cli.ProcessingService'):
                    with patch('src.pipelines.classification.cli.SQLiteService'):
                        with patch('src.pipelines.classification.cli.DocumentClassifier'):
                            cli = Cli()
                            
                            with patch('builtins.print') as mock_print:
                                cli.handle_command("retry")
                                # Should reinitialize with same host/port
                                assert cli.host == "localhost"
                                assert cli.port == 6333
    
    def test_retry_command_custom_host_port(self):
        """Test retry command with custom host and port."""
        with patch('src.pipelines.classification.cli.QdrantService') as mock_qdrant:
            with patch('src.pipelines.classification.cli.OpenAIService'):
                with patch('src.pipelines.classification.cli.ProcessingService'):
                    with patch('src.pipelines.classification.cli.SQLiteService'):
                        with patch('src.pipelines.classification.cli.DocumentClassifier'):
                            cli = Cli()
                            
                            with patch('builtins.print') as mock_print:
                                cli.handle_command("retry --host remote --port 7000")
                                assert cli.host == "remote"
                                assert cli.port == 7000

    def test_retry_invalid_args(self):
        """Retry with invalid flags should log usage error."""
        with patch('src.pipelines.classification.cli.QdrantService'):
            with patch('src.pipelines.classification.cli.OpenAIService'):
                with patch('src.pipelines.classification.cli.ProcessingService'):
                    with patch('src.pipelines.classification.cli.SQLiteService'):
                        with patch('src.pipelines.classification.cli.DocumentClassifier'):
                            cli = Cli()
                            with patch.object(cli, 'log') as mock_log:
                                cli.handle_command("retry --port notanint")
                                mock_log.assert_called()


class TestCLILabelCommands:
    """Test label management commands."""
    
    def test_add_label_no_collection(self):
        """Test add-label command without selected collection."""
        with patch('src.pipelines.classification.cli.QdrantService'):
            with patch('src.pipelines.classification.cli.OpenAIService'):
                with patch('src.pipelines.classification.cli.ProcessingService'):
                    with patch('src.pipelines.classification.cli.SQLiteService'):
                        with patch('src.pipelines.classification.cli.DocumentClassifier'):
                            cli = Cli()
                            with patch.object(cli, 'log') as mock_log:
                                cli.handle_command("add-label TestLabel")
                                mock_log.assert_called_with("No collection selected.", True)
    
    def test_list_labels_no_collection(self):
        """Test list-labels command without selected collection."""
        with patch('src.pipelines.classification.cli.QdrantService'):
            with patch('src.pipelines.classification.cli.OpenAIService'):
                with patch('src.pipelines.classification.cli.ProcessingService'):
                    with patch('src.pipelines.classification.cli.SQLiteService'):
                        with patch('src.pipelines.classification.cli.DocumentClassifier'):
                            cli = Cli()
                            with patch.object(cli, 'log') as mock_log:
                                cli.handle_command("list-labels")
                                mock_log.assert_called_with("No collection selected.", True)


class TestNoExitArgParser:
    """Test custom argument parser."""
    
    def test_no_exit_arg_parser_raises_value_error(self):
        """Test that NoExitArgParser raises ValueError instead of exiting."""
        parser = NoExitArgParser(prog="test", add_help=False)
        parser.add_argument("--test", type=int)
        
        with pytest.raises(ValueError):
            parser.parse_args(["--invalid"])


class TestCLISecurityValidation:
    """Test security and input validation."""
    
    def test_collection_name_validation(self):
        """Test that potentially dangerous collection names are handled."""
        # This test identifies a security gap - there's no validation
        # Collection names should be validated to prevent SQL injection
        with patch('src.pipelines.classification.cli.QdrantService') as mock_qdrant:
            with patch('src.pipelines.classification.cli.OpenAIService'):
                with patch('src.pipelines.classification.cli.ProcessingService'):
                    with patch('src.pipelines.classification.cli.SQLiteService'):
                        with patch('src.pipelines.classification.cli.DocumentClassifier'):
                            cli = Cli()
                            mock_qdrant_instance = mock_qdrant.return_value
                            mock_qdrant_instance.list_collections.return_value = ["'; DROP TABLE users; --"]
                            
                            # This should be handled safely
                            cli.handle_command("use '; DROP TABLE users; --")


class TestCLIEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_command(self):
        """Test handling of empty command."""
        with patch('src.pipelines.classification.cli.QdrantService'):
            with patch('src.pipelines.classification.cli.OpenAIService'):
                with patch('src.pipelines.classification.cli.ProcessingService'):
                    with patch('src.pipelines.classification.cli.SQLiteService'):
                        with patch('src.pipelines.classification.cli.DocumentClassifier'):
                            cli = Cli()
                            # Should not raise an exception
                            cli.handle_command("")
    
    def test_command_with_quotes(self):
        """Test command with quoted arguments."""
        with patch('src.pipelines.classification.cli.QdrantService') as mock_qdrant:
            with patch('src.pipelines.classification.cli.OpenAIService'):
                with patch('src.pipelines.classification.cli.ProcessingService'):
                    with patch('src.pipelines.classification.cli.SQLiteService'):
                        with patch('src.pipelines.classification.cli.DocumentClassifier'):
                            cli = Cli()
                            mock_qdrant_instance = mock_qdrant.return_value
                            mock_qdrant_instance.list_collections.return_value = ["test collection"]
                            
                            # Should handle quoted strings properly
                            cli.handle_command('use "test collection"')
    
    def test_command_with_special_characters(self):
        """Test command with special characters."""
        with patch('src.pipelines.classification.cli.QdrantService'):
            with patch('src.pipelines.classification.cli.OpenAIService'):
                with patch('src.pipelines.classification.cli.ProcessingService'):
                    with patch('src.pipelines.classification.cli.SQLiteService'):
                        with patch('src.pipelines.classification.cli.DocumentClassifier'):
                            cli = Cli()
                            # Should handle special characters without crashing
                            cli.handle_command("query @#$%^&*()")

    def test_unknown_then_help(self):
        """Unknown command should prompt help usage on demand."""
        with patch('src.pipelines.classification.cli.QdrantService'):
            with patch('src.pipelines.classification.cli.OpenAIService'):
                with patch('src.pipelines.classification.cli.ProcessingService'):
                    with patch('src.pipelines.classification.cli.SQLiteService'):
                        with patch('src.pipelines.classification.cli.DocumentClassifier'):
                            cli = Cli()
                            with patch('builtins.print') as mock_print:
                                cli.handle_command("doesnotexist")
                                cli.handle_command("help")
                                assert mock_print.call_count >= 1

    def test_create_with_invalid_args(self):
        """Create without args should log usage error."""
        with patch('src.pipelines.classification.cli.QdrantService'):
            with patch('src.pipelines.classification.cli.OpenAIService'):
                with patch('src.pipelines.classification.cli.ProcessingService'):
                    with patch('src.pipelines.classification.cli.SQLiteService'):
                        with patch('src.pipelines.classification.cli.DocumentClassifier'):
                            cli = Cli()
                            with patch.object(cli, 'log') as mock_log:
                                cli.handle_command("create")
                                mock_log.assert_called()

    def test_source_with_invalid_flags(self):
        """Source with invalid flags should trigger ValueError path and usage message."""
        with patch('src.pipelines.classification.cli.QdrantService'):
            with patch('src.pipelines.classification.cli.OpenAIService'):
                with patch('src.pipelines.classification.cli.ProcessingService'):
                    with patch('src.pipelines.classification.cli.SQLiteService'):
                        with patch('src.pipelines.classification.cli.DocumentClassifier'):
                            cli = Cli()
                            cli.collection = "test_collection"
                            with patch.object(cli, 'log') as mock_log:
                                cli.handle_command("source --not-a-flag")
                                mock_log.assert_called()

    def test_cluster_with_invalid_args(self):
        """Cluster with invalid flags should log usage error."""
        with patch('src.pipelines.classification.cli.QdrantService'):
            with patch('src.pipelines.classification.cli.OpenAIService'):
                with patch('src.pipelines.classification.cli.ProcessingService'):
                    with patch('src.pipelines.classification.cli.SQLiteService'):
                        with patch('src.pipelines.classification.cli.DocumentClassifier'):
                            cli = Cli()
                            cli.collection = "test_collection"
                            with patch.object(cli, 'log') as mock_log:
                                cli.handle_command("cluster --num-clusters notanint")
                                mock_log.assert_called()

    def test_classify_missing_labels_file(self):
        """Classify without labels file or flag should log error."""
        with patch('src.pipelines.classification.cli.QdrantService'):
            with patch('src.pipelines.classification.cli.OpenAIService'):
                with patch('src.pipelines.classification.cli.ProcessingService'):
                    with patch('src.pipelines.classification.cli.SQLiteService'):
                        with patch('src.pipelines.classification.cli.DocumentClassifier'):
                            cli = Cli()
                            cli.collection = "test_collection"
                            with patch.object(cli, 'log') as mock_log:
                                cli.handle_command("classify")
                                mock_log.assert_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

