import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[5]
POWERSHELL=shutil.which('pwsh') or shutil.which('powershell')


@unittest.skipUnless(POWERSHELL,'PowerShell is unavailable')
class SignEvaluationOrderTest(unittest.TestCase):
    def run_evaluation(self,fail_freeze=False):
        temporary=tempfile.TemporaryDirectory(dir=ROOT/'tmp');self.addCleanup(temporary.cleanup)
        package=Path(temporary.name)/'sign package';(package/'scripts').mkdir(parents=True)
        (package/'scripts/train.py').touch()
        trace=package/'calls.jsonl'
        (package/'scripts/run_training_tool.ps1').write_text('''param($Entry, [string[]]$ScriptArguments, $Workspace)
$path = Join-Path (Split-Path -Parent $PSScriptRoot) 'calls.jsonl'
ConvertTo-Json -InputObject @($ScriptArguments) -Compress | Add-Content -LiteralPath $path -Encoding utf8
$global:LASTEXITCODE = 0
'''+("if ($ScriptArguments[0] -eq 'freeze') { $global:LASTEXITCODE = 7 }\n" if fail_freeze else ''),encoding='utf8')
        metadata=package/'runs/sign/unit/metadata';metadata.mkdir(parents=True)
        (metadata/'artifacts.json').write_text(json.dumps({'best':'training/weights/best.pt'}),encoding='utf8')
        command=[POWERSHELL,'-NoProfile','-File',str(ROOT/'lab/features/sign/training/scripts/evaluate_sign.ps1'),
                 '-Root',str(package),'-Run','runs/sign/unit']
        result=subprocess.run(command,capture_output=True,text=True,encoding='utf8',timeout=30)
        return result,[json.loads(line) for line in trace.read_text(encoding='utf-8-sig').splitlines()]

    def test_validation_and_reports_precede_freezing_and_test(self):
        result,calls=self.run_evaluation()
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual([args[0] for args in calls],['evaluate','sign-report','freeze','evaluate','sign-report'])
        self.assertEqual(calls[0][calls[0].index('--split')+1],'val')
        self.assertEqual(calls[3][calls[3].index('--split')+1],'test')
        self.assertIn('frozen_selection',calls[3][calls[3].index('--weights')+1])

    def test_failed_freeze_prevents_test_access(self):
        result,calls=self.run_evaluation(True)
        self.assertEqual(result.returncode,7,result.stderr)
        self.assertEqual([args[0] for args in calls],['evaluate','sign-report','freeze'])


if __name__=='__main__':unittest.main()
