import yaml
from pathlib import Path


def load_config(config_file):
    """
    Load YAML config and expand all _dir keys:
    - plots_dir -> output_root
    - other _dir -> workflow root from base_dirs
    """
    with open(config_file, "r") as f:
        cfg = yaml.safe_load(f)

    base_dirs = cfg.get("base_dirs", {})

    def get_workflow_root(workflow_name):
        """Infer the workflow-specific root key from workflow name."""
        if workflow_name == "report":
            return "report_root"
        elif workflow_name.endswith("_emissions"):
            return workflow_name.replace("_emissions", "_root")
        else:
            return workflow_name + "_root"  # fallback

    def expand(node, workflow_key=None, current_key=None):
        if isinstance(node, dict):
            new_dict = {}
            for k, v in node.items():
                # Set workflow_key only for top-level workflow
                new_workflow = workflow_key
                if workflow_key is None:
                    potential_root = get_workflow_root(k)
                    if potential_root in base_dirs:
                        new_workflow = k
                
                new_dict[k] = expand(v, workflow_key=new_workflow, current_key=k)
            return new_dict

        elif isinstance(node, list):
            return [expand(v, workflow_key, current_key) for v in node]

        elif isinstance(node, str):
            p = Path(node)
            if p.is_absolute():
                return str(p)

            if workflow_key and current_key:
                # plots_dir always uses output_root
                if current_key in ["plots_dir", "json_dir"] and "output_root" in base_dirs:
                    return str(Path(base_dirs["output_root"]) / p)
                # any other _dir uses workflow root
                elif current_key.endswith("_dir"):
                    root_key = get_workflow_root(workflow_key)
                    if root_key in base_dirs:
                        return str(Path(base_dirs[root_key]) / p)
            return str(p)

        else:
            return node

    return expand(cfg)
