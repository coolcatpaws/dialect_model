import re
import subprocess

cleaned_lines = []
with open("requirements.txt", "r") as f:
    for line in f:
        # Match lines like: h11 @ file://...
        match = re.match(r"([a-zA-Z0-9_\-]+) @ file://", line)
        if match:
            pkg = match.group(1)
            try:
                # Get version info using pip show
                result = subprocess.run(
                    ["pip", "show", pkg], capture_output=True, text=True
                )
                version = None
                for l in result.stdout.splitlines():
                    if l.startswith("Version:"):
                        version = l.split(":", 1)[1].strip()
                        break
                if version:
                    cleaned_lines.append(f"{pkg}=={version}\n")
                    continue
            except Exception as e:
                print(f"Warning: Could not resolve version for {pkg}: {e}")
        cleaned_lines.append(line)

# Write to a clean version
with open("requirements.txt", "w") as f:
    f.writelines(cleaned_lines)

print("Cleaned requirements written to requirements.txt")
