TASK_TO_PROMPT = {
    "tha": {
        ####
        # NLU Tasks
        ####
        # Tasks.QUESTION_ANSWERING
        "QA": [
            "คำถาม: [QUESTION]\nตัวเลือก: [ANSWER_CHOICES]\nในบรรดา a, b, c, d, e ตัวเลือกที่ถูกต้องคือ: [LABEL_CHOICE]",
        ],
        # Tasks.MACHINE_TRANSLATION
        "MT": [
            "แปลข้อความต่อไปนี้จาก [SOURCE] เป็น [TARGET] ให้การแปลของคุณโดยตรงโดยไม่ต้องมีข้อมูลเพิ่มเติมใดๆ\nข้อความ: [INPUT]\nคำแปล:",
        ],
        "SUM": [
            "จงสรุปข้อความด้านล่าง\nข้อความ: [INPUT]\nสรุป:",
        ],
        # Tasks.INSTRUCTION_TUNING
        "IT": [
            "Task: [INPUT]\n คำตอบของคุณคืออะไร?",
        ],
        # Task.QA_EXTRACTIVE_ABSTRACTIVE
        "QAE": [
            "โปรดอ้างอิงถึงข้อความด้านล่างนี้และตอบคำถามต่อไปนี้ โดยตอบโดยใช้แค่ข้อความที่อยู่ในบทความ:\nข้อความ: [CONTEXT]\nคำถาม: [QUESTION]\nคำตอบ:",
        ],
    },
    "eng": {
        "QA": [
            "Question: [QUESTION]\nChoices: [ANSWER_CHOICES]\nAnswer: [LABEL_CHOICE]",
        ],
    },
}
