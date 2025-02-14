input_file = "reports/intraday/nsc_master.txt"
output_file = "reports/intraday/output.txt"

with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for line in infile:
        if line.startswith("NSE:"):
            parts = line.split()
            if len(parts) > 1:  # Ensure there are at least two parts
                outfile.write(parts[1] + "\n")  # Write second item to output file

print(f"Processed lines written to {output_file}")
