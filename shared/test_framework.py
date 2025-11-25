"""
Unified Testing Framework for Transfer Learning Course
Provides consistent testing across all lessons
"""

import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Tuple
from abc import ABC, abstractmethod

from .config import DEVICE_INFO, TEST_CONFIG, DATA_DIR
from .common import print_section_header, save_experiment_results


class TestResult:
    """Container for test results"""
    
    def __init__(self, name: str, passed: bool, message: str = "", 
                 duration: float = 0.0, details: Dict[str, Any] = None):
        self.name = name
        self.passed = passed
        self.message = message
        self.duration = duration
        self.details = details or {}
        self.timestamp = time.time()
    
    def __str__(self):
        status = "PASSED" if self.passed else "FAILED"
        duration_str = f"({self.duration:.3f}s)" if self.duration > 0 else ""
        message_str = f" - {self.message}" if self.message else ""
        return f"[{status}] {self.name} {duration_str}{message_str}"


class BaseTest(ABC):
    """Base class for all lesson tests"""
    
    def __init__(self, name: str, description: str = "", timeout: int = None):
        self.name = name
        self.description = description
        self.timeout = timeout or TEST_CONFIG["timeout"]["normal"]
    
    @abstractmethod
    def run(self) -> TestResult:
        """Run the test and return result"""
        pass
    
    def run_with_timeout(self) -> TestResult:
        """Run test with timeout protection"""
        start_time = time.time()
        
        try:
            result = self.run()
            duration = time.time() - start_time
            result.duration = duration
            return result
            
        except TimeoutError:
            duration = time.time() - start_time
            return TestResult(
                name=self.name,
                passed=False,
                message=f"Test timed out after {duration:.1f}s",
                duration=duration
            )
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                name=self.name,
                passed=False,
                message=f"Test failed with error: {str(e)}",
                duration=duration
            )


class EnvironmentTest(BaseTest):
    """Test environment setup and configuration"""
    
    def __init__(self):
        super().__init__("Environment Setup", "Verify Python environment and dependencies")
    
    def run(self) -> TestResult:
        """Test environment configuration"""
        issues = []
        
        # Test Python version
        import sys
        python_version = tuple(map(int, sys.version.split()[0].split('.')))
        if python_version < (3, 8):
            issues.append(f"Python {sys.version.split()[0]} < 3.8")
        
        # Test PyTorch installation
        try:
            import torch
            torch_version = torch.__version__
        except ImportError:
            issues.append("PyTorch not installed")
            return TestResult(self.name, False, "; ".join(issues))
        
        # Test torchvision
        try:
            import torchvision
        except ImportError:
            issues.append("Torchvision not installed")
        
        # Test data science libraries
        required_libs = ['numpy', 'matplotlib', 'pandas', 'seaborn', 'sklearn', 'PIL']
        for lib in required_libs:
            try:
                __import__(lib)
            except ImportError:
                issues.append(f"{lib} not installed")
        
        if issues:
            return TestResult(self.name, False, "; ".join(issues))
        
        details = {
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__
        }
        
        return TestResult(self.name, True, "All dependencies available", details=details)


class DeviceTest(BaseTest):
    """Test hardware device configuration"""
    
    def __init__(self):
        super().__init__("Hardware Device", "Test GPU/CPU device availability")
    
    def run(self) -> TestResult:
        """Test device configuration"""
        device_info = DEVICE_INFO
        device = device_info["device"]
        
        # Test basic tensor operations
        try:
            x = torch.randn(10, 10, device=device)
            y = torch.randn(10, 10, device=device)
            z = torch.mm(x, y)
            
            assert z.shape == (10, 10), f"Unexpected tensor shape: {z.shape}"
            assert z.device == device, f"Tensor on wrong device: {z.device}"
            
        except Exception as e:
            return TestResult(self.name, False, f"Device operation failed: {e}")
        
        details = {
            "device_type": device_info["type"],
            "device_name": device_info["name"],
            "device_description": device_info["description"]
        }
        
        if device_info.get("memory_gb"):
            details["memory_gb"] = device_info["memory_gb"]
        
        return TestResult(self.name, True, f"Device {device} working correctly", details=details)


class DataTest(BaseTest):
    """Test data loading and preprocessing"""
    
    def __init__(self, dataset_name: str = "flowers102"):
        super().__init__(f"Data Loading ({dataset_name})", f"Test {dataset_name} dataset access")
        self.dataset_name = dataset_name
    
    def run(self) -> TestResult:
        """Test data loading"""
        try:
            import torchvision.transforms as transforms
            from torchvision.datasets import ImageFolder
            
            # Check if data directory exists
            data_path = DATA_DIR / self.dataset_name
            if not data_path.exists():
                return TestResult(
                    self.name, False, 
                    f"Dataset directory not found: {data_path}"
                )
            
            # Try to create dataset
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
            
            dataset = ImageFolder(root=data_path / "train", transform=transform)
            
            if len(dataset) == 0:
                return TestResult(self.name, False, "Dataset is empty")
            
            # Test data loader
            data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=4, shuffle=False, num_workers=0
            )
            
            # Test loading one batch
            images, labels = next(iter(data_loader))
            
            assert images.shape[0] <= 4, f"Unexpected batch size: {images.shape[0]}"
            assert len(images.shape) == 4, f"Unexpected image tensor shape: {images.shape}"
            assert images.shape[1] == 3, f"Expected 3 channels, got: {images.shape[1]}"
            
            details = {
                "dataset_size": len(dataset),
                "num_classes": len(dataset.classes),
                "image_shape": list(images.shape[1:]),
                "sample_batch_size": images.shape[0]
            }
            
            return TestResult(
                self.name, True, 
                f"Dataset loaded successfully ({len(dataset)} samples, {len(dataset.classes)} classes)",
                details=details
            )
            
        except Exception as e:
            return TestResult(self.name, False, f"Data loading failed: {e}")


class ModelTest(BaseTest):
    """Test model creation and basic operations"""
    
    def __init__(self, model_name: str, num_classes: int = 102):
        super().__init__(f"Model Creation ({model_name})", f"Test {model_name} model instantiation")
        self.model_name = model_name
        self.num_classes = num_classes
    
    def run(self) -> TestResult:
        """Test model creation"""
        try:
            from .common import create_model_from_config, count_parameters
            
            # Create model
            model = create_model_from_config(self.model_name, self.num_classes)
            
            # Test model on device
            device = DEVICE_INFO["device"]
            model = model.to(device)
            
            # Test forward pass
            batch_size = 2
            dummy_input = torch.randn(batch_size, 3, 224, 224, device=device)
            
            with torch.no_grad():
                output = model(dummy_input)
            
            expected_shape = (batch_size, self.num_classes)
            assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
            
            # Count parameters
            param_count = count_parameters(model)
            
            details = {
                "model_name": self.model_name,
                "num_classes": self.num_classes,
                "output_shape": list(output.shape),
                "total_parameters": param_count["total"],
                "trainable_parameters": param_count["trainable"]
            }
            
            return TestResult(
                self.name, True,
                f"Model created successfully ({param_count['total']:,} parameters)",
                details=details
            )
            
        except Exception as e:
            return TestResult(self.name, False, f"Model creation failed: {e}")


class TrainingTest(BaseTest):
    """Test training functionality"""
    
    def __init__(self, model_name: str, quick_test: bool = True):
        super().__init__(f"Training Test ({model_name})", "Test basic training loop")
        self.model_name = model_name
        self.quick_test = quick_test
    
    def run(self) -> TestResult:
        """Test training functionality"""
        try:
            from .common import create_model_from_config
            import torch.optim as optim
            
            # Create model
            model = create_model_from_config(self.model_name, num_classes=5)  # Small test
            device = DEVICE_INFO["device"]
            model = model.to(device)
            
            # Create dummy data
            batch_size = 4 if self.quick_test else 16
            dummy_images = torch.randn(batch_size, 3, 224, 224, device=device)
            dummy_labels = torch.randint(0, 5, (batch_size,), device=device)
            
            # Setup training
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            
            model.train()
            
            # Training step
            optimizer.zero_grad()
            outputs = model(dummy_images)
            loss = criterion(outputs, dummy_labels)
            loss.backward()
            optimizer.step()
            
            # Validation step
            model.eval()
            with torch.no_grad():
                val_outputs = model(dummy_images)
                val_loss = criterion(val_outputs, dummy_labels)
            
            details = {
                "model_name": self.model_name,
                "training_loss": float(loss.item()),
                "validation_loss": float(val_loss.item()),
                "batch_size": batch_size,
                "output_shape": list(outputs.shape)
            }
            
            return TestResult(
                self.name, True,
                f"Training test successful (loss: {loss.item():.4f})",
                details=details
            )
            
        except Exception as e:
            return TestResult(self.name, False, f"Training test failed: {e}")


class PerformanceTest(BaseTest):
    """Test model performance and benchmarking"""
    
    def __init__(self, model_name: str):
        super().__init__(f"Performance Test ({model_name})", "Test model inference speed")
        self.model_name = model_name
    
    def run(self) -> TestResult:
        """Test model performance"""
        try:
            from .common import create_model_from_config, benchmark_model
            
            # Create and prepare model
            model = create_model_from_config(self.model_name, num_classes=102)
            device = DEVICE_INFO["device"]
            model = model.to(device)
            
            # Benchmark model
            benchmark_results = benchmark_model(
                model, input_size=(3, 224, 224), device=device, num_runs=50
            )
            
            mean_time = benchmark_results["mean_time_ms"]
            fps = benchmark_results["fps"]
            
            # Performance thresholds
            thresholds = TEST_CONFIG["performance_thresholds"]
            
            if mean_time > 1000:  # More than 1 second
                performance_level = "Slow"
            elif mean_time > 100:  # More than 100ms
                performance_level = "Moderate"
            else:
                performance_level = "Fast"
            
            details = {
                "model_name": self.model_name,
                "mean_inference_time_ms": mean_time,
                "fps": fps,
                "performance_level": performance_level,
                **benchmark_results
            }
            
            return TestResult(
                self.name, True,
                f"Performance: {mean_time:.1f}ms ({fps:.1f} FPS) - {performance_level}",
                details=details
            )
            
        except Exception as e:
            return TestResult(self.name, False, f"Performance test failed: {e}")


class LessonTestSuite:
    """Complete test suite for a lesson"""
    
    def __init__(self, lesson_name: str):
        self.lesson_name = lesson_name
        self.tests: List[BaseTest] = []
        self.results: List[TestResult] = []
    
    def add_test(self, test: BaseTest):
        """Add a test to the suite"""
        self.tests.append(test)
    
    def add_common_tests(self, include_models: List[str] = None):
        """Add common tests that every lesson should have"""
        self.add_test(EnvironmentTest())
        self.add_test(DeviceTest())
        self.add_test(DataTest())
        
        if include_models:
            for model_name in include_models:
                self.add_test(ModelTest(model_name))
    
    def run_all_tests(self, verbose: bool = True) -> Dict[str, Any]:
        """Run all tests in the suite"""
        print_section_header(f"Running {self.lesson_name} Test Suite")
        
        self.results = []
        start_time = time.time()
        
        for test in self.tests:
            if verbose:
                print(f"\nRunning: {test.name}")
                if test.description:
                    print(f"   Description: {test.description}")
            
            result = test.run_with_timeout()
            self.results.append(result)
            
            if verbose:
                print(f"   {result}")
        
        total_duration = time.time() - start_time
        
        # Generate summary
        passed_tests = [r for r in self.results if r.passed]
        failed_tests = [r for r in self.results if not r.passed]
        
        summary = {
            "lesson_name": self.lesson_name,
            "total_tests": len(self.results),
            "passed_tests": len(passed_tests),
            "failed_tests": len(failed_tests),
            "success_rate": len(passed_tests) / len(self.results) if self.results else 0,
            "total_duration": total_duration,
            "test_results": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "message": r.message,
                    "duration": r.duration,
                    "details": r.details
                } for r in self.results
            ]
        }
        
        self.print_summary(summary, verbose)
        
        # Save results
        save_experiment_results(f"tests_{self.lesson_name}", summary)
        
        return summary
    
    def print_summary(self, summary: Dict[str, Any], verbose: bool = True):
        """Print test summary"""
        print(f"\nTEST SUMMARY")
        print("=" * 50)
        
        passed = summary["passed_tests"]
        total = summary["total_tests"]
        success_rate = summary["success_rate"]
        
        print(f"Passed: {passed}/{total} tests")
        print(f"Success Rate: {success_rate:.1%}")
        print(f"Total Duration: {summary['total_duration']:.1f}s")
        
        if success_rate == 1.0:
            print(f"\nEXCELLENT! All tests passed!")
            print(f"{self.lesson_name} is working perfectly!")
        elif success_rate >= 0.8:
            print(f"\nGOOD! Most tests passed.")
            print(f"{summary['failed_tests']} test(s) need attention.")
        else:
            print(f"\nATTENTION! Multiple tests failed.")
            print(f"Please review failed tests and fix issues.")
        
        # List failed tests
        failed_results = [r for r in self.results if not r.passed]
        if failed_results and verbose:
            print(f"\nFAILED TESTS:")
            for result in failed_results:
                print(f"   - {result.name}: {result.message}")
        
        print("=" * 50)


# Export main components
__all__ = [
    'TestResult', 'BaseTest', 'LessonTestSuite',
    'EnvironmentTest', 'DeviceTest', 'DataTest', 
    'ModelTest', 'TrainingTest', 'PerformanceTest'
]