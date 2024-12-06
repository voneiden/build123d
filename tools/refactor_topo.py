"""
refactor topology

name: refactor_topology.py
by:   Gumyr
date: Dec 05, 2024

desc:
    This python script refactors the very large topology.py module into several
    files based on the topological heirarchical order:
    + shape_core.py - base classes Shape, ShapeList
    + utils.py - utility classes & functions
    + zero_d.py - Vertex
    + one_d.py - Mixin1D, Edge, Wire
    + two_d.py - Mixin2D, Face, Shell
    + three_d.py - Mixin3D, Solid
    + composite.py - Compound
    Each of these modules import lower order modules to avoid import loops. They
    also may contain functions used both by end users and higher order modules.

license:

    Copyright 2024 Gumyr

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

"""

from pathlib import Path
import libcst as cst
import libcst.matchers as m
from typing import List, Set, Dict
from rope.base.project import Project
from rope.refactor.importutils import ImportOrganizer
import subprocess
from datetime import datetime


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


class ClassMethodExtractor(cst.CSTVisitor):
    def __init__(self, methods_to_convert: List[str]):
        self.methods_to_convert = methods_to_convert
        self.extracted_methods: List[cst.FunctionDef] = []

    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        # Extract the class name to append it to the function name
        self.current_class_name = node.name.value
        self.generic_visit(node)  # Continue to visit child nodes

    def leave_ClassDef(self, original_node: cst.ClassDef) -> None:
        # Clear the current class name after leaving the class
        self.current_class_name = None

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        # Check if the function should be converted
        if node.name.value in self.methods_to_convert and self.current_class_name:
            # Rename the method by appending the class name to avoid conflicts
            new_name = f"{node.name.value}_{self.current_class_name.lower()}"
            renamed_node = node.with_changes(name=cst.Name(new_name))
            # Remove `self` from parameters since it's now a standalone function
            if renamed_node.params.params:
                renamed_node = renamed_node.with_changes(
                    params=renamed_node.params.with_changes(
                        params=renamed_node.params.params[1:]
                    )
                )
            self.extracted_methods.append(renamed_node)


def write_topo_class_files(
    source_tree: cst.Module,
    extracted_classes: Dict[str, cst.ClassDef],
    imports: Set[str],
    output_dir: Path,
) -> None:
    """Write files for each group of classes:"""
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sort imports for consistency
    imports_code = "\n".join(imports)

    # Describe where the functions should go
    function_source = {
        "shape_core": [
            "downcast",
            "fix",
            "get_top_level_topods_shapes",
            "_sew_topods_faces",
            "shapetype",
            "_topods_compound_dim",
            "_topods_entities",
            "_topods_face_normal_at",
            "apply_ocp_monkey_patches",
            "unwrap_topods_compound",
        ],
        "utils": [
            "delta",
            "_extrude_topods_shape",
            "find_max_dimension",
            "isclose_b",
            "_make_loft",
            "_make_topods_compound_from_shapes",
            "_make_topods_face_from_wires",
            "new_edges",
            "parse_arguments",
            "parse_kwargs",
            "polar",
            "_topods_bool_op",
            "tuplify",
            "unwrapped_shapetype",
        ],
        "zero_d": [
            "topo_explore_common_vertex",
        ],
        "one_d": [
            "edges_to_wires",
            "topo_explore_connected_edges",
        ],
        "two_d": ["sort_wires_by_build_order"],
    }

    # Define class groupings based on layers
    class_groups = {
        "utils": ["_ClassMethodProxy"],
        "shape_core": [
            "Shape",
            "Comparable",
            "ShapePredicate",
            "GroupBy",
            "ShapeList",
            "Joint",
            "SkipClean",
            "BoundBox",
        ],
        "zero_d": ["Vertex"],
        "one_d": ["Mixin1D", "Edge", "Wire"],
        "two_d": ["Mixin2D", "Face", "Shell"],
        "three_d": ["Mixin3D", "Solid"],
        "composite": ["Compound", "Curve", "Sketch", "Part"],
    }

    for group_name, class_names in class_groups.items():
        module_docstring = f"""
build123d topology

name: {group_name}.py
by:   Gumyr
date: {datetime.now().strftime('%B %d, %Y')}

desc:

license:

    Copyright {datetime.now().strftime('%Y')} Gumyr

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

"""
        header = [
            cst.SimpleStatementLine(
                [cst.Expr(cst.SimpleString(f'"""{module_docstring}"""'))]
            )
        ]

        if group_name in ["utils", "shape_core"]:
            function_collector = StandaloneFunctionAndVariableCollector()
            source_tree.visit(function_collector)

            variable_collector = GlobalVariableExtractor()
            source_tree.visit(variable_collector)

        group_classes = [
            extracted_classes[name] for name in class_names if name in extracted_classes
        ]
        if not group_classes:
            continue

        # Add imports for base classes based on layer dependencies
        additional_imports = []
        if group_name != "shape_core":
            additional_imports.append(
                "from .shape_core import Shape, ShapeList, BoundBox, SkipClean, TrimmingTool, Joint"
            )
            additional_imports.append("from .utils import _ClassMethodProxy")
        if group_name not in ["shape_core", "vertex"]:
            for sub_group_name in function_source.keys():
                additional_imports.append(
                    f"from .{sub_group_name} import "
                    + ",".join(function_source[sub_group_name])
                )
        if group_name not in ["shape_core", "utils", "vertex"]:
            additional_imports.append("from .zero_d import Vertex")
        if group_name in ["two_d"]:
            additional_imports.append("from .one_d import Mixin1D")

        if group_name in ["two_d", "three_d", "composite"]:
            additional_imports.append("from .one_d import Edge, Wire")
        if group_name in ["three_d", "composite"]:
            additional_imports.append("from .one_d import Mixin1D")

            additional_imports.append("from .two_d import Mixin2D, Face, Shell")
        if group_name == "composite":
            additional_imports.append("from .one_d import Mixin1D")
            additional_imports.append("from .three_d import Mixin3D, Solid")

        # Add TYPE_CHECKING imports
        if group_name not in ["composite"]:
            additional_imports.append("if TYPE_CHECKING: # pragma: no cover")
        if group_name in ["shape_core", "utils"]:
            additional_imports.append("    from .zero_d import Vertex")
        if group_name in ["shape_core", "utils", "zero_d"]:
            additional_imports.append("    from .one_d import Edge, Wire")
        if group_name in ["shape_core", "utils", "one_d"]:
            additional_imports.append("    from .two_d import Face, Shell")
        if group_name in ["shape_core", "utils", "one_d", "two_d"]:
            additional_imports.append("    from .three_d import Solid")
        if group_name in ["shape_core", "utils", "one_d", "two_d", "three_d"]:
            additional_imports.append(
                "    from .composite import Compound, Curve, Sketch, Part"
            )
        # Create class file (e.g., two_d.py)
        class_file = output_dir / f"{group_name}.py"
        all_imports_code = "\n".join([imports_code, *additional_imports])

        # if group_name in ["shape_core", "utils"]:
        if group_name in function_source.keys():
            body = [*cst.parse_module(all_imports_code).body]
            for func in function_collector.functions:
                if group_name == "shape_core" and func.name.value in [
                    "_topods_compound_dim",
                    "_topods_face_normal_at",
                    "apply_ocp_monkey_patches",
                ]:
                    body.append(func)

                # If this is the "apply_ocp_monkey_patches" function, add a call to it
                if func.name.value == "apply_ocp_monkey_patches":
                    apply_patches_call = cst.Expr(
                        value=cst.Call(func=cst.Name("apply_ocp_monkey_patches"))
                    )
                    body.append(apply_patches_call)
                    body.append(cst.EmptyLine(indent=False))
                    body.append(cst.EmptyLine(indent=False))

            if group_name == "shape_core":
                for var in variable_collector.global_variables:
                    # Check the name of the assigned variable(s)
                    for target in var.targets:
                        if isinstance(target.target, cst.Name):
                            var_name = target.target.value
                            # Check if the variable name is in the exclusion list
                            if var_name not in ["T", "K"]:
                                body.append(var)
                                body.append(cst.EmptyLine(indent=False))

            # Add classes and inject variables after a specific class
            for class_def in group_classes:
                body.append(class_def)

                # Inject variables after the specified class
                if class_def.name.value == "Comparable":
                    body.append(
                        cst.Comment(
                            "# This TypeVar allows IDEs to see the type of objects within the ShapeList"
                        )
                    )
                    body.append(cst.EmptyLine(indent=False))
                    for var in variable_collector.global_variables:
                        # Check the name of the assigned variable(s)
                        for target in var.targets:
                            if isinstance(target.target, cst.Name):
                                var_name = target.target.value
                                # Check if the variable name is in the inclusion list
                                if var_name in ["T", "K"]:
                                    body.append(var)
                                    body.append(cst.EmptyLine(indent=False))

            for func in function_collector.functions:
                if func.name.value in function_source[
                    group_name
                ] and func.name.value not in [
                    "_topods_compound_dim",
                    "_topods_face_normal_at",
                    "apply_ocp_monkey_patches",
                ]:
                    body.append(func)
            class_module = cst.Module(body=body, header=header)
        else:
            class_module = cst.Module(
                body=[*cst.parse_module(all_imports_code).body, *group_classes],
                header=header,
            )
        class_file.write_text(class_module.code)

        print(f"Created {class_file}")

    # Create __init__.py to make it a proper package
    # init_file = output_dir / "__init__.py"
    # init_content = []
    # for group_name in class_groups.keys():
    #     init_content.append(f"from .{group_name} import *")

    # init_file.write_text("\n".join(init_content))
    # print(f"Created {init_file}")


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
        subprocess.run(["black", file_path])

    else:
        print(f"No unused imports found in {file_path}")


class UnionToPipeTransformer(cst.CSTTransformer):
    def leave_Annotation(
        self, original_node: cst.Annotation, updated_node: cst.Annotation
    ) -> cst.Annotation:
        # Check if the annotation is using a Union
        if m.matches(updated_node.annotation, m.Subscript(value=m.Name("Union"))):
            subscript = updated_node.annotation
            if isinstance(subscript, cst.Subscript):
                elements = [elt.slice.value for elt in subscript.slice]
                # Build new binary operator nodes using | for each type in the Union
                new_annotation = elements[0]
                for element in elements[1:]:
                    new_annotation = cst.BinaryOperation(
                        left=new_annotation, operator=cst.BitOr(), right=element
                    )
                return updated_node.with_changes(annotation=new_annotation)
        return updated_node


def main():
    # Define paths
    script_dir = Path(__file__).parent
    topo_file = script_dir / ".." / "src" / "build123d" / "topology_old.py"
    output_dir = script_dir / ".." / "src" / "build123d" / "topology"
    topo_file = topo_file.resolve()
    output_dir = output_dir.resolve()

    # Define classes to extract
    class_names = [
        "_ClassMethodProxy",
        "BoundBox",
        "Shape",
        "Compound",
        "Solid",
        "Shell",
        "Face",
        "Wire",
        "Edge",
        "Vertex",
        "Curve",
        "Sketch",
        "Part",
        "Mixin1D",
        "Mixin2D",
        "Mixin3D",
        "Comparable",
        "ShapePredicate",
        "SkipClean",
        "ShapeList",
        "GroupBy",
        "Joint",
    ]

    # Parse source file and collect imports
    source_tree = cst.parse_module(topo_file.read_text())
    source_tree = source_tree.visit(UnionToPipeTransformer())
    # transformed_module = source_tree.visit(UnionToPipeTransformer())
    # print(transformed_module.code)

    collector = ImportCollector()
    source_tree.visit(collector)

    # Extract classes
    extractor = ClassExtractor(class_names)
    source_tree.visit(extractor)

    # Extract mixin classes
    mixin_extractor = MixinClassExtractor()
    source_tree.visit(mixin_extractor)

    # Extract functions
    function_collector = StandaloneFunctionAndVariableCollector()
    source_tree.visit(function_collector)
    # for f in function_collector.functions:
    #     print(f.name.value)

    # Write the class files
    write_topo_class_files(
        source_tree=source_tree,
        extracted_classes=extractor.extracted_classes,
        imports=collector.imports,
        output_dir=output_dir,
    )

    # Create a Rope project instance
    # project = Project(str(script_dir))
    project = Project(str(output_dir))

    # Clean up imports
    for file in output_dir.glob("*.py"):
        if file.name == "__init__.py":
            continue
        remove_unused_imports(file, project)


if __name__ == "__main__":
    main()
