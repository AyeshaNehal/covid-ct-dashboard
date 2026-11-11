import zipfile
from collections import defaultdict
from pathlib import Path

zip_path = Path("archive (10).zip")
print(f"Opening {zip_path} (size: {zip_path.stat().st_size / 1e9:.2f} GB)...")

with zipfile.ZipFile(zip_path, 'r') as z:
    names = z.namelist()
    print(f"Total files in archive: {len(names)}")
    
    # Analyze directory prefixes
    top_dirs = set()
    for name in names:
        parts = Path(name).parts
        if len(parts) > 1:
            top_dirs.add(parts[0])
            if len(parts) > 2:
                top_dirs.add(f"{parts[0]}/{parts[1]}")
    
    print("\nTop directories:")
    for d in sorted(top_dirs)[:30]:
        print(" ", d)

    # Check sample files per pattern
    print("\nSample paths:")
    for name in names[:20]:
        print(" ", name)
