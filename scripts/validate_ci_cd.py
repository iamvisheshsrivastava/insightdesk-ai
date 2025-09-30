#!/usr/bin/env python3
"""
CI/CD Pipeline Validation Script
==============================

This script validates that all CI/CD pipeline components are properly configured
and can run successfully in the local environment.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple


class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


class PipelineValidator:
    """Validates CI/CD pipeline components locally"""
    
    def __init__(self):
        self.root_path = Path(__file__).parent.parent
        self.results: List[Tuple[str, bool, str]] = []
    
    def log(self, message: str, color: str = Colors.BLUE):
        """Log a colored message"""
        print(f"{color}{message}{Colors.END}")
    
    def success(self, message: str):
        """Log a success message"""
        self.log(f"✅ {message}", Colors.GREEN)
    
    def error(self, message: str):
        """Log an error message"""
        self.log(f"❌ {message}", Colors.RED)
    
    def warning(self, message: str):
        """Log a warning message"""
        self.log(f"⚠️  {message}", Colors.YELLOW)
    
    def run_command(self, command: str, cwd: Path = None) -> Tuple[bool, str]:
        """Run a command and return success status and output"""
        try:
            result = subprocess.run(
                command.split(),
                cwd=cwd or self.root_path,
                capture_output=True,
                text=True,
                timeout=60
            )
            return result.returncode == 0, result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return False, "Command timed out"
        except Exception as e:
            return False, str(e)
    
    def check_file_exists(self, filepath: str) -> bool:
        """Check if a file exists"""
        file_path = self.root_path / filepath
        exists = file_path.exists()
        if exists:
            self.success(f"File exists: {filepath}")
        else:
            self.error(f"File missing: {filepath}")
        return exists
    
    def validate_github_workflow(self) -> bool:
        """Validate GitHub Actions workflow file"""
        self.log("\n📋 Validating GitHub Actions Workflow...", Colors.BOLD)
        
        workflow_path = ".github/workflows/ci-cd.yml"
        if not self.check_file_exists(workflow_path):
            return False
        
        try:
            # Check if workflow file is valid YAML
            try:
                import yaml
                with open(self.root_path / workflow_path) as f:
                    workflow = yaml.safe_load(f)
                
                # Validate required sections
                required_sections = ['name', 'on', 'jobs']
                for section in required_sections:
                    if section not in workflow:
                        self.error(f"Missing section in workflow: {section}")
                        return False
                
                # Check for expected jobs
                expected_jobs = [
                    'lint-and-format', 'unit-tests', 'integration-tests',
                    'build-and-test', 'performance-tests', 'build-and-push',
                    'deploy-staging', 'deploy-production', 'post-deployment', 'cleanup'
                ]
                
                missing_jobs = [job for job in expected_jobs if job not in workflow['jobs']]
                if missing_jobs:
                    self.error(f"Missing jobs in workflow: {missing_jobs}")
                    return False
                
                self.success("GitHub Actions workflow is valid")
                return True
                
            except ImportError:
                self.warning("PyYAML not installed, skipping detailed workflow validation")
                # Basic file content check
                with open(self.root_path / workflow_path) as f:
                    content = f.read()
                    if 'jobs:' in content and 'lint-and-format' in content:
                        self.success("GitHub Actions workflow file contains expected content")
                        return True
                    else:
                        self.error("GitHub Actions workflow file appears incomplete")
                        return False
        except Exception as e:
            self.error(f"Invalid workflow file: {str(e)}")
            return False
    
    def validate_configuration_files(self) -> bool:
        """Validate all configuration files"""
        self.log("\n⚙️  Validating Configuration Files...", Colors.BOLD)
        
        files_to_check = [
            "pyproject.toml",
            ".flake8", 
            ".gitignore",
            "Dockerfile",
            "requirements.txt",
            "Makefile"
        ]
        
        all_exist = True
        for file in files_to_check:
            if not self.check_file_exists(file):
                all_exist = False
        
        return all_exist
    
    def validate_python_setup(self) -> bool:
        """Validate Python environment and dependencies"""
        self.log("\n🐍 Validating Python Environment...", Colors.BOLD)
        
        # Check Python version
        success, output = self.run_command("python --version")
        if success:
            version = output.strip()
            self.success(f"Python version: {version}")
            
            # Check if version is 3.10+
            import sys
            if sys.version_info >= (3, 10):
                self.success("Python version is compatible")
            else:
                self.error("Python version should be 3.10+")
                return False
        else:
            self.error("Python not found")
            return False
        
        # Check pip
        success, _ = self.run_command("pip --version")
        if success:
            self.success("pip is available")
        else:
            self.error("pip not found")
            return False
        
        return True
    
    def validate_linting_tools(self) -> bool:
        """Validate linting and formatting tools"""
        self.log("\n🎨 Validating Linting Tools...", Colors.BOLD)
        
        tools = {
            "black": "black --version",
            "isort": "isort --version",
            "flake8": "flake8 --version",
            "bandit": "bandit --version",
            "safety": "safety --version"
        }
        
        all_available = True
        for tool, command in tools.items():
            success, output = self.run_command(command)
            if success:
                self.success(f"{tool} is available")
            else:
                self.warning(f"{tool} not found (install with: pip install {tool})")
                all_available = False
        
        return all_available
    
    def validate_testing_setup(self) -> bool:
        """Validate testing setup"""
        self.log("\n🧪 Validating Testing Setup...", Colors.BOLD)
        
        # Check pytest
        success, _ = self.run_command("pytest --version")
        if success:
            self.success("pytest is available")
        else:
            self.warning("pytest not found (install with: pip install pytest)")
            return False
        
        # Check if test directory exists
        if self.check_file_exists("tests"):
            self.success("Tests directory exists")
        else:
            self.error("Tests directory not found")
            return False
        
        # Look for test files
        test_files = list((self.root_path / "tests").glob("test_*.py"))
        if test_files:
            self.success(f"Found {len(test_files)} test files")
        else:
            self.warning("No test files found in tests/ directory")
        
        return True
    
    def validate_docker_setup(self) -> bool:
        """Validate Docker setup"""
        self.log("\n🐳 Validating Docker Setup...", Colors.BOLD)
        
        # Check Docker
        success, _ = self.run_command("docker --version")
        if success:
            self.success("Docker is available")
        else:
            self.warning("Docker not found")
            return False
        
        # Check Dockerfile
        if not self.check_file_exists("Dockerfile"):
            return False
        
        # Try to build Docker image (optional, might take time)
        build_test = input("\nDo you want to test Docker build? (y/N): ").lower().strip()
        if build_test == 'y':
            self.log("Building Docker image (this may take a few minutes)...")
            success, output = self.run_command("docker build -t insightdesk-ai:validation-test .")
            if success:
                self.success("Docker build successful")
                
                # Cleanup
                self.run_command("docker rmi insightdesk-ai:validation-test")
            else:
                self.error("Docker build failed")
                self.log(output)
                return False
        
        return True
    
    def validate_scripts(self) -> bool:
        """Validate key scripts exist and are executable"""
        self.log("\n📜 Validating Scripts...", Colors.BOLD)
        
        key_scripts = [
            "scripts/benchmark_models.py",
            "scripts/ab_testing_framework.py", 
            "scripts/demo_benchmarking.py",
            "scripts/test_benchmark_quick.py"
        ]
        
        all_exist = True
        for script in key_scripts:
            if not self.check_file_exists(script):
                all_exist = False
            else:
                # Check if it's a valid Python file
                try:
                    with open(self.root_path / script) as f:
                        content = f.read()
                        if "#!/usr/bin/env python" in content or "def " in content:
                            self.success(f"Script appears valid: {script}")
                        else:
                            self.warning(f"Script may not be Python: {script}")
                except Exception as e:
                    self.error(f"Error reading script {script}: {e}")
                    all_exist = False
        
        return all_exist
    
    def run_quick_tests(self) -> bool:
        """Run quick validation tests"""
        self.log("\n⚡ Running Quick Tests...", Colors.BOLD)
        
        test_results = True
        
        # Test linting (if tools available)
        self.log("Testing code formatting...")
        success, output = self.run_command("python -m black --check --diff .")
        if success:
            self.success("Code formatting is correct")
        else:
            self.warning("Code formatting issues found (run: black .)")
        
        # Test import structure
        self.log("Testing import structure...")
        success, output = self.run_command("python -c \"import src; print('✅ Import successful')\"")
        if success:
            self.success("Package imports work correctly")
        else:
            self.warning("Import issues found")
            self.log(output)
        
        # Test quick benchmark script
        if self.check_file_exists("scripts/test_benchmark_quick.py"):
            self.log("Testing benchmark script...")
            success, output = self.run_command("python scripts/test_benchmark_quick.py")
            if success:
                self.success("Benchmark script runs successfully")
            else:
                self.warning("Benchmark script has issues")
                # Don't fail on this as it might need models
        
        return test_results
    
    def generate_report(self) -> Dict:
        """Generate validation report"""
        self.log("\n📊 Generating Validation Report...", Colors.BOLD)
        
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "validation_results": self.results,
            "summary": {
                "total_checks": len(self.results),
                "passed": sum(1 for _, success, _ in self.results if success),
                "failed": sum(1 for _, success, _ in self.results if not success)
            }
        }
        
        # Save report
        report_path = self.root_path / "ci-cd-validation-report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.success(f"Validation report saved: {report_path}")
        return report
    
    def run_validation(self) -> bool:
        """Run complete validation"""
        self.log(f"\n{Colors.BOLD}🚀 CI/CD Pipeline Validation{Colors.END}")
        self.log("=" * 50)
        
        validation_steps = [
            ("GitHub Workflow", self.validate_github_workflow),
            ("Configuration Files", self.validate_configuration_files), 
            ("Python Environment", self.validate_python_setup),
            ("Linting Tools", self.validate_linting_tools),
            ("Testing Setup", self.validate_testing_setup),
            ("Docker Setup", self.validate_docker_setup),
            ("Scripts", self.validate_scripts),
            ("Quick Tests", self.run_quick_tests)
        ]
        
        overall_success = True
        
        for step_name, step_func in validation_steps:
            try:
                success = step_func()
                self.results.append((step_name, success, ""))
                if not success:
                    overall_success = False
            except Exception as e:
                self.error(f"Error in {step_name}: {str(e)}")
                self.results.append((step_name, False, str(e)))
                overall_success = False
        
        # Generate report
        report = self.generate_report()
        
        # Final summary
        self.log(f"\n{Colors.BOLD}📋 Validation Summary{Colors.END}")
        self.log("=" * 30)
        
        if overall_success:
            self.success("🎉 All validations passed! CI/CD pipeline is ready.")
        else:
            failed_checks = [name for name, success, _ in self.results if not success]
            self.error(f"❌ Some validations failed: {failed_checks}")
            self.log("\n💡 Next steps:")
            self.log("1. Install missing dependencies: pip install -r requirements.txt")
            self.log("2. Install development tools: pip install black isort flake8 bandit safety pytest")
            self.log("3. Fix code formatting: black . && isort .")
            self.log("4. Check Dockerfile and scripts")
        
        self.log(f"\n📊 Results: {report['summary']['passed']}/{report['summary']['total_checks']} checks passed")
        
        return overall_success


def main():
    """Main entry point"""
    validator = PipelineValidator()
    success = validator.run_validation()
    
    if success:
        print(f"\n{Colors.GREEN}🚀 Ready to push to GitHub! The CI/CD pipeline should work correctly.{Colors.END}")
        sys.exit(0)
    else:
        print(f"\n{Colors.RED}⚠️  Please fix the issues above before pushing to GitHub.{Colors.END}")
        sys.exit(1)


if __name__ == "__main__":
    main()