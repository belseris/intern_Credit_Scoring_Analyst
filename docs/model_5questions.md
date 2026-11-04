# T01: 5 questions about the model (Bell)

Questions from running `reference_solution.py` (Taiwan credit cards) and `th_altdata_scorecard.py` (synthetic Thai top-up data). Each question names the step in the script output it comes from.

## 1. ตัวแปรเดียวคุมทั้งโมเดล (reference_solution.py ขั้น 3.4 และ 5.5)

pay_status_1 มี IV 0.861 เกินเกณฑ์ 0.5 ที่ต้องเช็ก และในตารางคะแนนห่างกันถึง 71 แต้ม (51–122) ขณะที่ตัวอื่นห่างกันแค่ 5–16 แต้ม โมเดลพึ่งตัวแปรเดียวมากขนาดนี้ยอมรับได้ไหม และนับเป็น leakage หรือเปล่า

English: pay_status_1 has IV 0.861 and a 71-point spread, while other features spread only 5–16 points. Is it acceptable for one variable to drive the model this much, and is it close to leakage?

## 2. ความนิ่งที่ดูดีเกินจริง (reference_solution.py ขั้น 6.3)

PSI = 0.002 แต่ train กับ test สุ่มมาจากช่วงเวลาเดียวกัน ตัวเลขนี้จึงยังพิสูจน์ความนิ่งข้ามเวลาไม่ได้ ถ้าข้อมูลไม่มีวันที่ ทีมเช็กความนิ่งกันยังไง

English: PSI is 0.002, but train and test come from the same period. How does the team check stability when the data has no dates?

## 3. สเกลคะแนนไม่ตรงกัน (reference_solution.py ขั้น 5.4 เทียบกับชีตงาน T09)

สคริปต์ใช้ PDO 20, 600 = odds 50:1 แต่งาน T09 กำหนด 600 = 30:1 และ 40 แต้มต่อการเพิ่ม odds 2 เท่า สเกลไหนคือมาตรฐานของทีม ถ้าเปลี่ยน จุดตัด Red 518.8 และ Amber 533.6 ต้องเปลี่ยนตามด้วยใช่ไหม

English: The script uses PDO 20 and 600 = 50:1, but T09 uses 600 = 30:1 with 40 points to double. Which scale is the team standard? Our Red and Amber cut-offs would change.

## 4. ตัวเลขไหนเชื่อได้จากข้อมูลสังเคราะห์ (th_altdata_scorecard.py ขั้น 6.1)

ข้อมูลสังเคราะห์มาจากกติกาที่เราตั้งเอง AUC ที่ได้จึงสะท้อนค่าในตัวสร้างข้อมูล ควรตั้งให้ทำนายได้แรงแค่ไหน (R1 ได้ราว 0.62) และผลส่วนไหนพอเชื่อได้บ้าง เช่น ใช้แค่เช็กว่าขั้นตอนทำงานถูก

English: The synthetic AUC reflects the generator settings. What signal strength should we assume (R1 is about 0.62), and which results can we trust, for example only that the pipeline works?

## 5. positive-only ใช้เดี่ยวๆ ไม่ได้ (th_altdata_scorecard.py ขั้น 6.5)

แบบ positive-only ทำให้ไม่มีใครตกกลุ่ม Red เพราะข้อมูลมือถือเป็นแหล่งเดียว ถ้าจะใช้จริงควรรวมกับคะแนนบูโรแบบไหน และทีมอยากให้ข้อมูลมือถือหักคะแนนลูกค้าได้ไหม

English: With positive-only, nobody can be Red because phone data is the only source. How should it combine with a bureau score, and should phone data be allowed to lower a score?
