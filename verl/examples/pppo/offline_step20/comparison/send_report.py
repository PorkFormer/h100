"""Send the user-authorized comparison bundle via the existing mail account."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
from email.message import EmailMessage
from email.utils import formatdate, make_msgid

ROOT = Path(__file__).resolve().parents[5]
REPORT = ROOT / 'reports/pppo20_vs_ncbr_h2048_l8192_s300_20261114'
RECIPIENT = '3531125084@qq.com'


def main():
    attachment = REPORT / 'report_bundle.zip'
    digest = hashlib.sha256(attachment.read_bytes()).hexdigest()
    receipt_path = REPORT / 'delivery_receipt.json'
    if receipt_path.exists():
        prior = json.loads(receipt_path.read_text())
        if prior.get('smtp_accepted') and prior.get('attachment_sha256') == digest:
            print('Already SMTP accepted for this exact bundle.')
            return
    message = EmailMessage()
    message['From'] = RECIPIENT
    message['To'] = RECIPIENT
    message['Subject'] = 'PPPO Step20 与 NCBR H2048/L8192 Step300：训练动态、GPU-hours与离线评测完整报告'
    message['Date'] = formatdate(localtime=False, usegmt=True)
    message['Message-ID'] = make_msgid(domain='qq.com')
    message.set_content('''本次只比较 PPPO Step20 与 NCBR H=2048/L=8192 Step300。

训练进程口径：PPPO 641.941 GPU-hours，NCBR 628.650 GPU-hours，PPPO高2.114%。
逐步训练+在线验证计时口径：637.537 vs 614.377 GPU-hours。二者口径在附件中分别解释。
PPPO完成20步、实测320次optimizer更新；独立holdout从9.77%升至35.55%，eta保持15%。
PPPO续写阶段占step计时94.26%，actor更新占1.51%；前缀token节省不能直接解释为GPU-hours节省。

统一H_eval=8192，每题32次采样；下列为PPPO / NCBR：
AIME24 Avg@32 9.90% / 16.56%，Pass@32 36.67% / 46.67%。
AIME25 Avg@32 7.81% / 14.48%，Pass@32 36.67% / 40.00%。
AMC23 Avg@32 39.61% / 69.84%，Pass@32 90.00% / 95.00%。
MATH500 Avg@32 51.93% / 59.03%，Pass@32 87.00% / 70.40%。

PPPO单次正确率均较低，MATH500的32次尝试覆盖率更高。近似GPU预算不代表相同步数、样本曝光或训练协议。
附件含中文报告、三张图表、两组完整训练配置（含NCBR恢复前后）、逐步CSV/JSONL、开销分解、离线指标/标准误差/配对区间、来源SHA256及生成脚本。
请解压后打开REPORT_ZH.html或REPORT_ZH.md。训练仍暂停在Step20，本次未恢复。
''')
    message.add_attachment(attachment.read_bytes(), maintype='application', subtype='zip', filename=attachment.name)
    receipt = dict(recipient=RECIPIENT, message_id=message['Message-ID'],
                   attempted_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   attachment_sha256=digest, attachment_bytes=attachment.stat().st_size,
                   smtp_accepted=False)
    try:
        result = subprocess.run(['/usr/bin/msmtp', '-t'], input=message.as_bytes(), capture_output=True, timeout=60)
        receipt.update(exit_code=result.returncode, smtp_accepted=result.returncode == 0)
    except Exception as error:
        receipt['error_type'] = type(error).__name__
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(receipt, ensure_ascii=False))
    if not receipt['smtp_accepted']:
        raise SystemExit('SMTP did not confirm acceptance; inspect transport before retrying.')


if __name__ == '__main__':
    main()
