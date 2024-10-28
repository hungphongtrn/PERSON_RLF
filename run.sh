#!/bin/bash

# Check if a file is provided as argument
if [ $# -ne 1 ]; then
    echo "Usage: $0 <command_file>"
    exit 1
fi

command_file="$1"

# Check if the file exists
if [ ! -f "$command_file" ]; then
    echo "Error: File '$command_file' not found"
    exit 1
fi

# Initialize counters
total_commands=0
successful_commands=0
failed_commands=0

# Read and execute each line
while IFS= read -r cmd || [ -n "$cmd" ]; do
    # Skip empty lines and comments
    if [ -z "$cmd" ] || [[ "$cmd" =~ ^[[:space:]]*# ]]; then
        continue
    fi

    total_commands=$((total_commands + 1))
    echo "Executing command ($total_commands): $cmd"
    
    # Execute the command and capture its exit status
    eval "$cmd"
    exit_status=$?
    
    if [ $exit_status -eq 0 ]; then
        echo "Command completed successfully"
        successful_commands=$((successful_commands + 1))
    else
        echo "Command failed with exit status: $exit_status"
        failed_commands=$((failed_commands + 1))
    fi
    
    echo "----------------------------------------"
done < "$command_file"

# Print summary
echo "Execution Summary:"
echo "Total commands: $total_commands"
echo "Successful: $successful_commands"
echo "Failed: $failed_commands"
