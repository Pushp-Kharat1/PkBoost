import os
import glob

def extract_rust_code_to_file(root_dir, output_file="all_rust_code.txt"):
    """
    Extract all Rust code from .rs files into a single text file.
    
    Args:
        root_dir (str): Root directory to search for Rust files
        output_file (str): Output file name
    """
    
    # Find all .rs files recursively
    rust_files = glob.glob(os.path.join(root_dir, "**", "*.rs"), recursive=True)
    
    # Exclude target directory if it exists
    rust_files = [f for f in rust_files if "target" not in f]
    
    print(f"Found {len(rust_files)} Rust files:")
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.write("=== COMPLETE RUST CODE EXTRACTION ===\n")
        outfile.write(f"Total files: {len(rust_files)}\n")
        outfile.write("=" * 50 + "\n\n")
        
        for i, file_path in enumerate(sorted(rust_files), 1):
            relative_path = os.path.relpath(file_path, root_dir)
            
            outfile.write(f"\n{'='*60}\n")
            outfile.write(f"FILE {i}: {relative_path}\n")
            outfile.write(f"{'='*60}\n\n")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as infile:
                    content = infile.read()
                    outfile.write(content)
                    outfile.write("\n")  # Add spacing between files
                    
                print(f"  [{i}/{len(rust_files)}] Processed: {relative_path}")
                
            except Exception as e:
                print(f"  [!] Error reading {relative_path}: {e}")
                outfile.write(f"// ERROR READING FILE: {e}\n")
    
    print(f"\n✅ Extraction complete! All code saved to: {output_file}")
    print(f"📁 Total files processed: {len(rust_files)}")

def extract_with_file_structure(root_dir, output_file="rust_code_with_structure.txt"):
    """
    Alternative version that shows the file structure more clearly.
    """
    
    rust_files = glob.glob(os.path.join(root_dir, "**", "*.rs"), recursive=True)
    rust_files = [f for f in rust_files if "target" not in f]
    
    # Group files by directory
    files_by_dir = {}
    for file_path in rust_files:
        dir_name = os.path.dirname(file_path)
        if dir_name not in files_by_dir:
            files_by_dir[dir_name] = []
        files_by_dir[dir_name].append(file_path)
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.write("=== RUST PROJECT STRUCTURE AND CODE ===\n\n")
        
        for dir_path, files in sorted(files_by_dir.items()):
            # Write directory header
            rel_dir = os.path.relpath(dir_path, root_dir)
            outfile.write(f"📁 DIRECTORY: {rel_dir}\n")
            outfile.write("-" * 40 + "\n")
            
            for file_path in sorted(files):
                relative_path = os.path.relpath(file_path, root_dir)
                outfile.write(f"\n📄 FILE: {relative_path}\n")
                outfile.write("─" * 50 + "\n\n")
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        content = infile.read()
                        outfile.write(content)
                        outfile.write("\n\n")
                        
                except Exception as e:
                    outfile.write(f"// ERROR READING FILE: {e}\n\n")
        
        outfile.write("=" * 50 + "\n")
        outfile.write(f"END OF EXTRACTION - {len(rust_files)} files total\n")

if __name__ == "__main__":
    # Set your Rust project root directory here
    project_root = input("Enter the path to your Rust project root: ").strip()
    
    if not project_root:
        # Default to current directory if nothing entered
        project_root = "."
    
    if not os.path.exists(project_root):
        print(f"Error: Path '{project_root}' does not exist!")
        exit(1)
    
    print(f"Scanning directory: {os.path.abspath(project_root)}")
    
    # Create both versions
    extract_rust_code_to_file(project_root, "all_rust_code.txt")
    extract_with_file_structure(project_root, "rust_code_with_structure.txt")
    
    print("\n📊 Summary:")
    print("  - all_rust_code.txt: Simple concatenation of all files")
    print("  - rust_code_with_structure.txt: Organized by directory structure")