"""Verify that paper Test evaluation cannot precede successful Val and freeze."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[5]
SHELL=shutil.which('pwsh') or shutil.which('powershell')


@unittest.skipUnless(SHELL,'PowerShell unavailable')
class PaperEvaluationOrderTest(unittest.TestCase):
    def run_evaluation(self,fail=None):
        temporary=tempfile.TemporaryDirectory(dir=ROOT/'tmp');self.addCleanup(temporary.cleanup)
        root=Path(temporary.name)/'paper package';(root/'scripts').mkdir(parents=True)
        (root/'scripts/train.py').touch()
        (root/'scripts/run_training_tool.ps1').write_text('''param($Entry,[string[]]$ScriptArguments,$Workspace)
ConvertTo-Json -InputObject @($ScriptArguments) -Compress | Add-Content -LiteralPath (Join-Path (Split-Path -Parent $PSScriptRoot) 'calls.jsonl') -Encoding utf8
$global:LASTEXITCODE=0
'''+(f"if ($ScriptArguments[0] -eq '{fail}') {{ $global:LASTEXITCODE=7 }}\n" if fail else ''),encoding='utf8')
        path=root/'runs/building_name_paper/unit/metadata';path.mkdir(parents=True)
        (path/'artifacts.json').write_text(json.dumps({'best':'training/weights/best.pt'}),encoding='utf8')
        result=subprocess.run([SHELL,'-NoProfile','-File',str(ROOT/'lab/features/fire/training/scripts/evaluate_building_name_paper.ps1'),
            '-Root',str(root),'-Run','runs/building_name_paper/unit'],capture_output=True,text=True,encoding='utf8',timeout=30)
        trace=root/'calls.jsonl'
        return result,[json.loads(line) for line in trace.read_text(encoding='utf-8-sig').splitlines()] if trace.exists() else []

    def test_full_val_reports_freeze_then_test_reports(self):
        result,calls=self.run_evaluation();self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual([c[0] for c in calls],['evaluate','paper-report','paper-sizes','freeze','evaluate','paper-report','paper-sizes'])
        self.assertEqual(calls[0][calls[0].index('--split')+1],'val')
        self.assertEqual(calls[4][calls[4].index('--split')+1],'test')
        self.assertIn('frozen_selection',calls[4][calls[4].index('--weights')+1])

    def test_failed_validation_report_or_freeze_prevents_test_access(self):
        for failure in ('paper-report','freeze'):
            result,calls=self.run_evaluation(failure);self.assertEqual(result.returncode,7,result.stderr)
            self.assertEqual(sum(c[0]=='evaluate' for c in calls),1)


if __name__=='__main__':unittest.main()
