#!/usr/bin/env python3
"""
Example script demonstrating how to use the DockerConfigExtractor 
from swebench_context_aware.py for getting Docker configurations from poly dataset.
"""

import sys
import os
from pathlib import Path

# Add the source directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from minisweagent.run.extra.swebench_context_aware import DockerConfigExtractor


def main():
    """Demonstrate Docker configuration extraction capabilities."""
    
    # Example 1: Extract from sample poly data
    extension_root = Path(__file__).resolve().parents[2]
    poly_data_dir = Path(os.environ.get("TRAJ_POLY_DIR", str(extension_root / "traj_poly")))
    
    # Sample instance IDs from different project types
    sample_instances = [
        {"instance_id": "sveltejs__svelte-728", "expected_type": "javascript"},
        {"instance_id": "prettier__prettier-8046", "expected_type": "javascript"},
        {"instance_id": "serverless__serverless-7374", "expected_type": "serverless"},
        {"instance_id": "mui__material-ui-11451", "expected_type": "javascript"},
    ]
    
    print("🐳 Docker Configuration Extractor Demo\n")
    
    for instance in sample_instances:
        instance_id = instance["instance_id"]
        print(f"📋 Processing instance: {instance_id}")
        
        # Get Docker configuration 
        docker_config = DockerConfigExtractor.get_docker_config_for_instance(
            instance, 
            poly_data_dir
        )
        
        print(f"   Source: {docker_config['source']}")
        if docker_config.get('dockerfile_content'):
            lines = docker_config['dockerfile_content'].split('\n')
            print(f"   Dockerfile preview (first 3 lines):")
            for i, line in enumerate(lines[:3]):
                print(f"     {i+1}: {line}")
            if len(lines) > 3:
                print(f"     ... ({len(lines)-3} more lines)")
        elif docker_config.get('base_image'):
            print(f"   Base image: {docker_config['base_image']}")
        print()

    # Example 2: Generate PolyBench Dockerfiles
    print("🏗️  Generated PolyBench Dockerfiles:\n")
    
    # JavaScript project
    js_dockerfile = DockerConfigExtractor.generate_polybench_dockerfile("javascript")
    print("JavaScript project Dockerfile:")
    print(js_dockerfile)
    print()
    
    # Python project  
    py_dockerfile = DockerConfigExtractor.generate_polybench_dockerfile("python")
    print("Python project Dockerfile:")
    print(py_dockerfile)
    print()
    
    # Custom JavaScript with different Node version
    custom_js = DockerConfigExtractor.generate_polybench_dockerfile(
        "javascript", 
        node_version="20.16.0",
        custom_commands=["RUN npm pkg set scripts.lint=\"echo noop\""]
    )
    print("Custom JavaScript (Node 20.16.0) Dockerfile:")
    print(custom_js)
    print()

    # Example 3: Extract Dockerfile from patch string
    print("🔍 Extract from patch string:\n")
    
    sample_patch = """diff --git a/Dockerfile b/Dockerfile
new file mode 100644
index 0000000..38ca023
--- /dev/null
+++ b/Dockerfile
@@ -0,0 +1,5 @@
+FROM polybench_javascript_base
+WORKDIR /testbed
+COPY . .
+RUN . /usr/local/nvm/nvm.sh && nvm use 16.20.2 && rm -rf node_modules && npm pkg set scripts.lint="echo noop" && npm install
+RUN . $NVM_DIR/nvm.sh && nvm alias default 16.20.2 && nvm use default
diff --git a/package.json b/package.json
index ded4c54..851a8bd 100644
"""
    
    extracted_dockerfile = DockerConfigExtractor.extract_dockerfile_from_patch(sample_patch)
    if extracted_dockerfile:
        print("Extracted Dockerfile:")
        print(extracted_dockerfile)
    else:
        print("No Dockerfile content found in patch")
        
    print("\n✅ Demo completed!")


if __name__ == "__main__":
    main()
