"""Execute evaluations after an independently verified Step20 pause; no monitor."""
import json
import sys
import time

import evaluate as evaluation


def notifier():
    import prepare
    prepare.RUN = evaluation.RUN
    sys.path.insert(0, str(evaluation.REPO / 'verl/examples/pppo'))
    import continue_ncbr_matched as mail

    def body(stage):
        status = evaluation.read(evaluation.RUN / 'training_status.json')
        lines = [stage, time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
                 'PPPO Step20 离线评测；评测后保持暂停，不自动恢复训练。',
                 'H_eval=8192，每题32次采样，沿用历史 prompt / seed / verifier。',
                 'PPPO: AIME24、AIME25、AMC23、MATH500；四组Step300基线: MATH500。',
                 '训练状态: ' + str(status.get('state')),
                 '最后成功step: ' + str(status.get('last_successful_step')),
                 '输出目录: ' + str(evaluation.OUT)]
        for arm in ('diagnostic', *evaluation.ARMS):
            path = evaluation.OUT / arm / 'status.json'
            record = evaluation.read(path) if path.exists() else {'status': 'NOT RUN'}
            lines.append(arm + ': ' + record['status'])
            scores = evaluation.OUT / arm / 'scores.json'
            if record['status'] == 'PASS' and scores.exists():
                lines.append(json.dumps(evaluation.read(scores)['metrics'], ensure_ascii=False))
        return '\n'.join(lines) + '\n'

    mail.RUN = evaluation.RUN
    mail.DIRECTORY = evaluation.RUN / 'automation'
    mail.status_body = body
    return mail.notify


if __name__ == '__main__':
    notifier()
    evaluation.run_all()
