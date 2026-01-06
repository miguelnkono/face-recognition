#!/usr/bin/env python3

import os
import sys
import argparse
from pathlib import Path
from typing import List
import shutil

class ImageRenamer:
    def __init__(self):
        self.supported_extensions = {
            '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif',
            '.gif', '.webp', '.svg', '.ico', '.jfif'
        }

    def get_image_files(self, folder_path: str) -> List[Path]:
        """Get all image files from a folder"""
        folder = Path(folder_path)

        if not folder.exists():
            print(f"❌ Error: Folder '{folder_path}' does not exist!")
            return []

        if not folder.is_dir():
            print(f"❌ Error: '{folder_path}' is not a directory!")
            return []

        # Get all image files
        image_files = []
        for ext in self.supported_extensions:
            image_files.extend(folder.glob(f"*{ext}"))
            image_files.extend(folder.glob(f"*{ext.upper()}"))

        # Sort files for consistent ordering
        image_files.sort()
        return image_files

    def rename_images(self,
                     folder_path: str,
                     base_name: str,
                     start_number: int = 1,
                     padding: int = 0,
                     preview_only: bool = False,
                     keep_original: bool = False,
                     extension: str = None) -> dict:
        """
        Rename all images in folder

        Args:
            folder_path: Path to folder containing images
            base_name: Base name for renamed files
            start_number: Starting number for sequence
            padding: Zero padding (e.g., 3 -> 001, 002, etc.)
            preview_only: Only show what would be renamed
            keep_original: Keep original files (copy instead of move)
            extension: Force specific extension (optional)

        Returns:
            Dictionary with rename results
        """
        image_files = self.get_image_files(folder_path)

        if not image_files:
            print(f"⚠️ No image files found in '{folder_path}'")
            print(f"   Supported extensions: {', '.join(sorted(self.supported_extensions))}")
            return {"total": 0, "renamed": 0, "errors": 0}

        print(f"📁 Found {len(image_files)} image file(s) in '{folder_path}'")

        if preview_only:
            print("\n🔍 PREVIEW MODE - No files will be changed")
        else:
            print("\n🚀 RENAMING FILES")

        print("=" * 60)

        results = {
            "total": len(image_files),
            "renamed": 0,
            "errors": 0,
            "operations": []
        }

        counter = start_number

        for i, old_path in enumerate(image_files, 1):
            try:
                # Get original extension or use specified one
                if extension:
                    new_ext = f".{extension.lstrip('.')}"
                else:
                    new_ext = old_path.suffix.lower()

                # Generate new filename with padding
                if padding > 0:
                    number_str = str(counter).zfill(padding)
                else:
                    number_str = str(counter)

                new_filename = f"{base_name}_{number_str}{new_ext}"
                new_path = old_path.parent / new_filename

                # Handle naming conflicts
                conflict_count = 0
                while new_path.exists():
                    conflict_count += 1
                    new_filename = f"{base_name}_{number_str}_{conflict_count}{new_ext}"
                    new_path = old_path.parent / new_filename

                # Store operation info
                operation = {
                    "old_name": old_path.name,
                    "new_name": new_filename,
                    "old_path": str(old_path),
                    "new_path": str(new_path)
                }

                if preview_only:
                    print(f"[{i:3d}] {old_path.name:30} → {new_filename:30} (Preview)")
                else:
                    # Perform the rename/copy
                    if keep_original:
                        shutil.copy2(old_path, new_path)
                        action = "Copied"
                    else:
                        old_path.rename(new_path)
                        action = "Renamed"

                    print(f"[{i:3d}] ✓ {old_path.name:30} → {new_filename:30} ({action})")
                    results["renamed"] += 1

                results["operations"].append(operation)
                counter += 1

            except Exception as e:
                print(f"[{i:3d}] ✗ {old_path.name:30} → ERROR: {str(e)}")
                results["errors"] += 1

        return results

    def interactive_mode(self):
        """Interactive mode for user-friendly operation"""
        print("🖼️  IMAGE BATCH RENAMER - INTERACTIVE MODE")
        print("=" * 50)

        # Get folder path
        while True:
            folder_path = input("\n📁 Enter folder path containing images: ").strip()
            if not folder_path:
                print("❌ Please enter a folder path")
                continue

            folder_path = os.path.expanduser(folder_path)
            if not os.path.exists(folder_path):
                print(f"❌ Folder '{folder_path}' does not exist!")
                continue

            break

        # Get base name
        while True:
            base_name = input("🏷️  Enter base name for files (e.g., 'vacation', 'portrait'): ").strip()
            if not base_name:
                print("❌ Please enter a base name")
                continue

            # Clean base name (remove invalid characters)
            import re
            base_name = re.sub(r'[<>:"/\\|?*]', '_', base_name)
            break

        # Get start number
        start_number = 1
        start_input = input("🔢 Enter starting number [1]: ").strip()
        if start_input:
            try:
                start_number = int(start_input)
                if start_number < 0:
                    print("⚠️  Using default start number 1")
                    start_number = 1
            except ValueError:
                print("⚠️  Using default start number 1")

        # Get padding
        padding = 0
        padding_input = input("0️⃣  Enter zero padding (e.g., 3 for 001, 002) [0]: ").strip()
        if padding_input:
            try:
                padding = int(padding_input)
                if padding < 0:
                    padding = 0
            except ValueError:
                padding = 0

        # Get extension preference
        extension = None
        ext_input = input("📄 Force specific extension (e.g., 'jpg', 'png') [keep original]: ").strip().lower()
        if ext_input:
            extension = ext_input

        # Get operation mode
        keep_original = False
        mode_input = input("💾 Keep original files? (y/N): ").strip().lower()
        if mode_input in ['y', 'yes']:
            keep_original = True

        # Preview first
        print("\n" + "=" * 50)
        print("🔍 PREVIEWING CHANGES:")
        print("=" * 50)

        preview_results = self.rename_images(
            folder_path=folder_path,
            base_name=base_name,
            start_number=start_number,
            padding=padding,
            preview_only=True,
            keep_original=keep_original,
            extension=extension
        )

        if preview_results["total"] == 0:
            return

        # Ask for confirmation
        confirm = input("\n✅ Proceed with rename? (y/N): ").strip().lower()

        if confirm in ['y', 'yes']:
            print("\n" + "=" * 50)
            print("🚀 PERFORMING RENAME OPERATION:")
            print("=" * 50)

            results = self.rename_images(
                folder_path=folder_path,
                base_name=base_name,
                start_number=start_number,
                padding=padding,
                preview_only=False,
                keep_original=keep_original,
                extension=extension
            )

            print("\n" + "=" * 50)
            print("📊 RENAME COMPLETE!")
            print("=" * 50)
            print(f"   Total files: {results['total']}")
            print(f"   Successfully processed: {results['renamed']}")
            print(f"   Errors: {results['errors']}")

            # Save log file
            log_file = os.path.join(folder_path, f"rename_log_{base_name}.txt")
            self.save_rename_log(log_file, results)
            print(f"   Log saved to: {log_file}")

        else:
            print("\n❌ Operation cancelled.")

    def save_rename_log(self, log_file: str, results: dict):
        """Save rename operations to a log file"""
        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write(f"Image Rename Log - {Path(log_file).parent.name}\n")
                f.write(f"Date: {self.get_current_timestamp()}\n")
                f.write(f"Base name: {results.get('base_name', 'N/A')}\n")
                f.write(f"Total files: {results['total']}\n")
                f.write(f"Successfully processed: {results['renamed']}\n")
                f.write(f"Errors: {results['errors']}\n")
                f.write("\n" + "=" * 60 + "\n")
                f.write("DETAILED OPERATIONS:\n")
                f.write("=" * 60 + "\n\n")

                for i, op in enumerate(results['operations'], 1):
                    f.write(f"[{i:3d}] {op['old_name']} → {op['new_name']}\n")

                f.write("\n" + "=" * 60 + "\n")
                f.write("END OF LOG\n")

            return True
        except Exception as e:
            print(f"⚠️  Could not save log file: {e}")
            return False

    def get_current_timestamp(self):
        """Get current timestamp for log file"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def restore_from_log(self, log_file: str):
        """Restore original names from log file"""
        if not os.path.exists(log_file):
            print(f"❌ Log file '{log_file}' not found!")
            return False

        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            operations = []
            in_operations = False

            for line in lines:
                line = line.strip()
                if "DETAILED OPERATIONS:" in line:
                    in_operations = True
                    continue
                if "END OF LOG" in line:
                    break

                if in_operations and line.startswith('['):
                    # Parse operation line: "[  1] old.jpg → new.jpg"
                    parts = line.split('→')
                    if len(parts) == 2:
                        old_part = parts[0].split(']')[1].strip()
                        new_name = parts[1].strip()
                        operations.append((old_part, new_name))

            if not operations:
                print("❌ No operations found in log file!")
                return False

            print(f"\n🔙 RESTORING FROM LOG: {log_file}")
            print(f"   Found {len(operations)} operations to restore")
            print("=" * 60)

            folder = Path(log_file).parent
            restored = 0
            errors = 0

            for old_name, new_name in operations:
                old_path = folder / old_name
                new_path = folder / new_name

                if new_path.exists() and old_path.exists():
                    # Both exist, can't restore
                    print(f"⚠️  Both '{old_name}' and '{new_name}' exist, skipping")
                    errors += 1
                elif new_path.exists():
                    # Restore original name
                    try:
                        new_path.rename(old_path)
                        print(f"✓ Restored: {new_name} → {old_name}")
                        restored += 1
                    except Exception as e:
                        print(f"✗ Error restoring {new_name}: {e}")
                        errors += 1
                else:
                    print(f"⚠️  File not found: {new_name}")
                    errors += 1

            print("\n" + "=" * 60)
            print(f"📊 RESTORE COMPLETE!")
            print(f"   Restored: {restored}")
            print(f"   Errors: {errors}")

            return True

        except Exception as e:
            print(f"❌ Error reading log file: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Batch rename images in a folder with sequential numbering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --name vacation --folder ./photos
  %(prog)s --name portrait --folder ./images --start 10 --padding 3
  %(prog)s --name product --folder ./pics --keep --preview
  %(prog)s --interactive
  %(prog)s --restore rename_log_vacation.txt
        """
    )

    # Main arguments
    parser.add_argument("--folder", "-f", help="Folder containing images")
    parser.add_argument("--name", "-n", help="Base name for renamed files")
    parser.add_argument("--start", "-s", type=int, default=1,
                       help="Starting number (default: 1)")
    parser.add_argument("--padding", "-p", type=int, default=0,
                       help="Zero padding length (e.g., 3 for 001, 002)")
    parser.add_argument("--extension", "-e",
                       help="Force specific extension (e.g., 'jpg', 'png')")

    # Mode flags
    parser.add_argument("--preview", action="store_true",
                       help="Preview changes without renaming")
    parser.add_argument("--keep", "-k", action="store_true",
                       help="Keep original files (copy instead of move)")
    parser.add_argument("--interactive", "-i", action="store_true",
                       help="Run in interactive mode")
    parser.add_argument("--restore", "-r",
                       help="Restore original names from log file")

    args = parser.parse_args()

    renamer = ImageRenamer()

    # Handle restore mode
    if args.restore:
        renamer.restore_from_log(args.restore)
        return

    # Handle interactive mode
    if args.interactive or (not args.folder and not args.name):
        renamer.interactive_mode()
        return

    # Check required arguments for batch mode
    if not args.folder:
        print("❌ Error: Folder path is required!")
        print("   Use --folder /path/to/images or --interactive mode")
        return

    if not args.name:
        print("❌ Error: Base name is required!")
        print("   Use --name base_name or --interactive mode")
        return

    # Validate folder exists
    if not os.path.exists(args.folder):
        print(f"❌ Error: Folder '{args.folder}' does not exist!")
        return

    # Perform rename
    print(f"🖼️  BATCH IMAGE RENAMER")
    print("=" * 50)
    print(f"   Folder: {args.folder}")
    print(f"   Base name: {args.name}")
    print(f"   Start number: {args.start}")
    print(f"   Padding: {args.padding}")
    print(f"   Extension: {args.extension or 'keep original'}")
    print(f"   Preview only: {args.preview}")
    print(f"   Keep originals: {args.keep}")
    print("=" * 50)

    results = renamer.rename_images(
        folder_path=args.folder,
        base_name=args.name,
        start_number=args.start,
        padding=args.padding,
        preview_only=args.preview,
        keep_original=args.keep,
        extension=args.extension
    )

    # Summary
    if not args.preview and results["total"] > 0:
        print("\n" + "=" * 50)
        print("📊 RENAME COMPLETE!")
        print("=" * 50)
        print(f"   Total files: {results['total']}")
        print(f"   Successfully processed: {results['renamed']}")
        print(f"   Errors: {results['errors']}")

        # Save log file
        if results['renamed'] > 0 and not args.preview:
            log_file = os.path.join(args.folder, f"rename_log_{args.name}.txt")
            # Add base name to results for log
            results["base_name"] = args.name
            renamer.save_rename_log(log_file, results)
            print(f"   Log saved to: {log_file}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Operation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
