from pathlib import Path
import libcst as cst
from typing import List, Set, Dict, Union
from pprint import pprint
from rope.base.project import Project
from rope.refactor.importutils import ImportOrganizer


class ImportCollector(cst.CSTVisitor):
    def __init__(self):
        self.imports: Set[str] = set()

    def visit_Import(self, node: cst.Import) -> None:
        for name in node.names:
            # Create a proper statement line
            stmt = cst.SimpleStatementLine([node])
            self.imports.add(cst.Module([stmt]).code)

    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        # Create a proper statement line
        stmt = cst.SimpleStatementLine([node])
        self.imports.add(cst.Module([stmt]).code)


class ClassExtractor(cst.CSTVisitor):
    def __init__(self, class_names_to_extract: List[str]):
        self.class_names = class_names_to_extract
        self.extracted_classes: Dict[str, cst.ClassDef] = {}

    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        if node.name.value in self.class_names:
            self.extracted_classes[node.name.value] = node


class MixinClassExtractor(cst.CSTVisitor):
    def __init__(self):
        self.extracted_classes: Dict[str, cst.ClassDef] = {}

    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        if "Mixin" in node.name.value:
            self.extracted_classes[node.name.value] = node


class StandaloneFunctionAndVariableCollector(cst.CSTVisitor):
    def __init__(self):
        self.functions: List[cst.FunctionDef] = []
        self.current_scope_level = 0  # Track nesting level

    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        # Entering a new class scope, increase nesting level
        self.current_scope_level += 1

    def leave_ClassDef(self, original_node: cst.ClassDef) -> None:
        if self.current_scope_level > 0:
            self.current_scope_level -= 1

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        if self.current_scope_level == 0:
            self.functions.append(node)


class GlobalVariableExtractor(cst.CSTVisitor):
    def __init__(self):
        # Store the global variable assignments
        self.global_variables: List[cst.Assign] = []

    def visit_Module(self, node: cst.Module) -> None:
        # Visit all assignments at the module level
        for statement in node.body:
            if isinstance(statement, cst.SimpleStatementLine):
                for assign in statement.body:
                    if isinstance(assign, cst.Assign):
                        self.global_variables.append(assign)


def write_topo_class_files(
    extracted_classes: Dict[str, cst.ClassDef],
    imports: Set[str],
    output_dir: Path,
) -> None:
    """
    Write files for each group of classes:
    1. Separate modules for "Shape", "Compound", "Solid", "Face" + "Shell", "Edge" + "Wire", and "Vertex"
    2. "ShapeList" is extracted into its own module and imported by all modules except "Shape"
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sort imports for consistency
    imports_code = "\n".join(imports)

    # Define class groupings based on layers
    class_groups = {
        "shape": ["Shape"],
        "vertex": ["Vertex"],
        "edge_wire": ["Mixin1D", "Edge", "Wire"],
        "face_shell": ["Face", "Shell"],
        "solid": ["Mixin3D", "Solid"],
        "compound": ["Compound"],
        "shape_list": ["ShapeList"],
    }

    # Write ShapeList class separately
    if "ShapeList" in extracted_classes:
        class_file = output_dir / "shape_list.py"
        shape_list_class = extracted_classes["ShapeList"]
        shape_list_module = cst.Module(
            body=[*cst.parse_module(imports_code).body, shape_list_class]
        )
        class_file.write_text(shape_list_module.code)
        print(f"Created {class_file}")

    for group_name, class_names in class_groups.items():
        if group_name == "shape_list":
            continue

        group_classes = [
            extracted_classes[name] for name in class_names if name in extracted_classes
        ]
        if not group_classes:
            continue

        # Add imports for base classes based on layer dependencies
        additional_imports = ["from .utils import *"]
        if group_name != "shape":
            additional_imports.append("from .shape import Shape")
            additional_imports.append("from .shape_list import ShapeList")
        if group_name in ["edge_wire", "face_shell", "solid", "compound"]:
            additional_imports.append("from .vertex import Vertex")
        if group_name in ["face_shell", "solid", "compound"]:
            additional_imports.append("from .edge_wire import Edge, Wire")
        if group_name in ["solid", "compound"]:
            additional_imports.append("from .face_shell import Face, Shell")
        if group_name == "compound":
            additional_imports.append("from .solid import Solid")

        # Create class file (e.g., face_shell.py)
        class_file = output_dir / f"{group_name}.py"
        all_imports_code = "\n".join([imports_code, *additional_imports])
        class_module = cst.Module(
            body=[*cst.parse_module(all_imports_code).body, *group_classes]
        )
        class_file.write_text(class_module.code)
        print(f"Created {class_file}")

    # Create __init__.py to make it a proper package
    init_file = output_dir / "__init__.py"
    init_content = []
    for group_name in class_groups.keys():
        if group_name != "shape_list":
            init_content.append(f"from .{group_name} import *")

    init_file.write_text("\n".join(init_content))
    print(f"Created {init_file}")


def write_utils_file(
    source_tree: cst.Module, imports: Set[str], output_dir: Path
) -> None:
    """
    Extract and write standalone functions and global variables to a utils.py file.

    Args:
        source_tree: The parsed source tree
        imports: Set of import statements
        output_dir: Directory to write the utils file
    """
    # Collect standalone functions and global variables
    function_collector = StandaloneFunctionAndVariableCollector()
    source_tree.visit(function_collector)

    variable_collector = GlobalVariableExtractor()
    source_tree.visit(variable_collector)

    # Create utils file
    utils_file = output_dir / "utils.py"

    # Prepare the module body
    module_body = []

    # Add imports
    imports_tree = cst.parse_module("\n".join(sorted(imports)))
    module_body.extend(imports_tree.body)

    # Add global variables with newlines
    for var in variable_collector.global_variables:
        module_body.append(var)
        module_body.append(cst.EmptyLine(indent=False))

    # Add a newline between variables and functions
    if variable_collector.global_variables and function_collector.functions:
        module_body.append(cst.EmptyLine(indent=False))

    # Add functions
    module_body.extend(function_collector.functions)

    # Create the module
    utils_module = cst.Module(body=module_body)

    # Write the file
    utils_file.write_text(utils_module.code)
    print(f"Created {utils_file}")


def remove_unused_imports(file_path: Path, project: Project) -> None:
    """Remove unused imports from a Python file using rope.

    Args:
        file_path: Path to the Python file to clean imports
        project: Rope project instance to refresh and use for cleaning
    """
    # Get the relative file path from the project root
    relative_path = file_path.relative_to(project.address)

    # Refresh the project to recognize new files
    project.validate()

    # Get the resource (file) to work on
    resource = project.get_resource(str(relative_path))

    # Create import organizer
    import_organizer = ImportOrganizer(project)

    # Get and apply the changes
    changes = import_organizer.organize_imports(resource)
    if changes:
        changes.do()
        print(f"Cleaned imports in {file_path}")
    else:
        print(f"No unused imports found in {file_path}")


def main():
    # Define paths
    script_dir = Path(__file__).parent
    topo_file = script_dir / "topology.py"
    output_dir = script_dir / "topology"

    # Define classes to extract
    class_names = [
        "Shape",
        "Compound",
        "Solid",
        "Shell",
        "Face",
        "Wire",
        "Edge",
        "Vertex",
        "Mixin0D",
        "Mixin1D",
        "Mixin2D",
        "Mixin3D",
        "MixinCompound",
        "ShapeList",
    ]

    # Parse source file and collect imports
    source_tree = cst.parse_module(topo_file.read_text())
    collector = ImportCollector()
    source_tree.visit(collector)

    # Extract classes
    extractor = ClassExtractor(class_names)
    source_tree.visit(extractor)

    # Extract mixin classes
    mixin_extractor = MixinClassExtractor()
    source_tree.visit(mixin_extractor)

    # Write the class files
    write_topo_class_files(
        extracted_classes=extractor.extracted_classes,
        imports=collector.imports,
        output_dir=output_dir,
    )

    # Write the utils file
    write_utils_file(
        source_tree=source_tree, imports=collector.imports, output_dir=output_dir
    )

    # Create a Rope project instance
    project = Project(str(script_dir))

    # Clean up imports
    for file in output_dir.glob("*.py"):
        remove_unused_imports(file, project)


if __name__ == "__main__":
    main()
