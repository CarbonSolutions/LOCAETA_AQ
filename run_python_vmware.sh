#!/bin/bash

# Define the path to the VMX file for the VM
VMX_PATH="/Users/yunhalee/Virtual Machines.localized/Windows 11 64-bit Arm.vmwarevm/Windows 11 64-bit Arm.vmx"

# Define the guest OS user credentials
GUEST_USER="defaultuser0"
GUEST_PASSWORD="LEE=1234"

# Define the paths to the Python interpreter and the Python script within the guest VM
PYTHON_INTERPRETER="C:\\LOCAETA\\miniconda3\\envs\\benmap\\python.exe"
PYTHON_SCRIPT="C:\\LOCAETA\\BenMAP\\batchmode\\run_all_benmap_modes.py"

# Run the Python script in the VM using vmrun
/Applications/VMware\ Fusion.app/Contents/Library/vmrun \
-T fusion -gu "$GUEST_USER" -gp "$GUEST_PASSWORD" \
runProgramInGuest "$VMX_PATH" "$PYTHON_INTERPRETER" "$PYTHON_SCRIPT"
