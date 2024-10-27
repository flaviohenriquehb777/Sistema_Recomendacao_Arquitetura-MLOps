#!/usr/bin/env python3
"""
Script direto para reescrever datas dos commits
Usa git reset e git commit --amend para alterar cada commit
"""

import subprocess
import sys
from datetime import datetime, timedelta
import os

def run_command(cmd, check=True):
    """Executa comando e retorna resultado"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=check)
        return result.stdout.strip(), result.stderr.strip()
    except subprocess.CalledProcessError as e:
        print(f"Erro ao executar: {cmd}")
        print(f"Stderr: {e.stderr}")
        return None, e.stderr

def get_professional_dates():
    """Retorna lista de datas profissionais distribuídas"""
    base_date = datetime(2024, 10, 15, 10, 0, 0)
    dates = []
    
    # Distribuir 50 commits ao longo de 12 meses
    for i in range(50):
        # Adicionar dias de forma não linear (mais commits no início e fim)
        if i < 15:  # Primeiros commits - outubro/novembro 2024
            days_offset = i * 3
        elif i < 35:  # Meio do projeto - dezembro 2024 a agosto 2025
            days_offset = 45 + (i - 15) * 8
        else:  # Commits finais - setembro/outubro 2025
            days_offset = 205 + (i - 35) * 4
        
        # Adicionar variação de horário
        hours_offset = (i % 8) + 9  # Entre 9h e 16h
        minutes_offset = (i * 17) % 60  # Minutos variados
        
        commit_date = base_date + timedelta(days=days_offset, hours=hours_offset-10, minutes=minutes_offset)
        dates.append(commit_date.strftime("%Y-%m-%d %H:%M:%S"))
    
    return dates

def rewrite_history_direct():
    """Reescreve histórico alterando datas diretamente"""
    print("Obtendo lista de commits...")
    
    # Obter commits do mais antigo para o mais novo
    stdout, stderr = run_command('git log --pretty=format:"%H|%s" --reverse')
    if not stdout:
        print("❌ Erro ao obter commits")
        return False
    
    commits = []
    for line in stdout.split('\n'):
        if '|' in line:
            hash_commit, message = line.split('|', 1)
            commits.append((hash_commit.strip(), message.strip()))
    
    print(f"Encontrados {len(commits)} commits")
    
    # Obter datas profissionais
    dates = get_professional_dates()
    
    if len(commits) != len(dates):
        print(f"⚠️ Ajustando para {len(commits)} commits")
        dates = dates[:len(commits)]
    
    print("Iniciando reescrita...")
    
    # Fazer checkout do primeiro commit
    first_commit = commits[0][0]
    stdout, stderr = run_command(f"git checkout {first_commit}")
    if stderr and "detached HEAD" not in stderr:
        print(f"❌ Erro no checkout: {stderr}")
        return False
    
    # Criar nova branch
    run_command("git checkout -b new-history")
    
    # Processar cada commit
    for i, ((commit_hash, message), new_date) in enumerate(zip(commits, dates)):
        print(f"Processando commit {i+1}/{len(commits)}: {message[:50]}...")
        
        if i == 0:
            # Primeiro commit - apenas alterar data
            cmd = f'git commit --amend --no-edit --date="{new_date}"'
            stdout, stderr = run_command(cmd, check=False)
            if stderr and "nothing to commit" not in stderr.lower():
                print(f"⚠️ Aviso no primeiro commit: {stderr}")
        else:
            # Commits subsequentes - cherry-pick e alterar data
            stdout, stderr = run_command(f"git cherry-pick {commit_hash}", check=False)
            if stderr and ("conflict" in stderr.lower() or "error" in stderr.lower()):
                # Resolver conflitos automaticamente
                run_command("git add .", check=False)
                run_command(f'git -c core.editor=true cherry-pick --continue', check=False)
            
            # Alterar data do commit
            cmd = f'git commit --amend --no-edit --date="{new_date}"'
            run_command(cmd, check=False)
    
    print("✅ Reescrita concluída!")
    return True

def main():
    """Função principal"""
    print("=== REESCRITA DIRETA DE DATAS ===")
    print("Alterando datas para cronograma profissional")
    print("Outubro 2024 - Outubro 2025")
    print()
    
    # Verificar repositório
    stdout, stderr = run_command("git status", check=False)
    if stderr:
        print("❌ Não é um repositório Git válido")
        sys.exit(1)
    
    # Verificar mudanças pendentes
    stdout, _ = run_command("git status --porcelain")
    if stdout:
        print("❌ Há mudanças não commitadas")
        sys.exit(1)
    
    # Backup
    print("Criando backup...")
    run_command("git branch -D backup-direct", check=False)
    current_branch, _ = run_command("git branch --show-current")
    run_command(f"git checkout -b backup-direct")
    run_command(f"git checkout {current_branch}")
    print("✅ Backup criado em 'backup-direct'")
    
    try:
        if rewrite_history_direct():
            print("\n🎉 SUCESSO!")
            print("Nova branch 'new-history' criada com cronograma profissional!")
            print("\nPróximos passos:")
            print("1. Verificar: git log --pretty=format:'%ad %s' --date=short")
            print("2. Substituir main: git checkout main && git reset --hard new-history")
            print("3. Enviar: git push --force-with-lease origin main")
        else:
            print("❌ Falha na reescrita")
            
    except Exception as e:
        print(f"❌ Erro: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()