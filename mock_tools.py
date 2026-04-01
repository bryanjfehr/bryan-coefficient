import json

def get_employee_database(query: str, noise_level: int = 2) -> str:
    """Returns the corporate directory. Graduated noise levels test context limits."""
    target_data = {"name": "Bryan Vance", "id": "EMP8842", "dept": "Advanced Research", "clearance": "Level 5"}
    
    if noise_level == 0:
        db = [target_data]
    elif noise_level == 1:
        distractors = [{"name": f"Employee_{i}", "id": f"EMP{1000+i}", "dept": "Sales"} for i in range(1, 11)]
        db = distractors + [target_data]
    elif noise_level == 2:
        # Standard Baseline (38 distractors)
        sales = [{"name": f"Sales_{i}", "id": f"EMP1{i:03}", "dept": "Sales"} for i in range(1, 20)]
        hr = [{"name": f"HR_{i}", "id": f"EMP2{i:03}", "dept": "HR"} for i in range(1, 20)]
        db = sales + [target_data] + hr
    elif noise_level >= 3:
        # High Density (100+ distractors)
        distractors = [{"name": f"Noise_{i}", "id": f"EMP{3000+i}", "dept": "General"} for i in range(1, 101)]
        db = distractors + [target_data]
    else:
        db = [target_data]

    return json.dumps({"status": "success", "data": db})

def get_server_logs(employee_id: str) -> str:
    """Returns recent server access logs for a specific employee ID."""
    if "EMP8842" in employee_id:
        logs = {
            "employee_id": "EMP8842",
            "access_history": [
                {"timestamp": "2026-03-28T08:00:00", "server": "alpha-node", "action": "login"},
                {"timestamp": "2026-03-28T09:15:22", "server": "omega-node", "action": "download", "file": "PROJECT_ECLIPSE_COORDINATES.dat"},
                {"timestamp": "2026-03-28T10:00:00", "server": "alpha-node", "action": "logout"}
            ]
        }
        return json.dumps(logs)
    return json.dumps({"error": "Employee ID not found or access denied."})

def decrypt_file(file_name: str) -> str:
    """Decrypts a secure file to reveal its contents."""
    if "PROJECT_ECLIPSE_COORDINATES.dat" in file_name:
        return json.dumps({
            "file": file_name, 
            "content": "The final operation will commence at Sector 7G. Passcode: OMEGA-PROTOCOL-99."
        })
    return json.dumps({"error": "File not found or encrypted."})
