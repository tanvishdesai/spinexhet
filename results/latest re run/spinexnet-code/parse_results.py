import re, ast

with open(r'c:\Users\DELL\Desktop\code_playground\het-spine\results-logs-3.txt', encoding='utf-8') as f:
    text = f.read()

pattern = re.compile(r"\{'epoch': \d+, 'train_total'[^}]+\}")
spinexnet = []
convnext = []
for m in pattern.finditer(text):
    try:
        d = ast.literal_eval(m.group())
        ep = d['epoch']
        c = d.get('train_concept', 0)
        wll = d['val_weighted_log_loss']
        ba = d['val_balanced_accuracy']
        f1 = d['val_macro_f1']
        auc = d['val_auc_ovr']
        if c > 0.01:
            spinexnet.append((ep, c, wll, ba, f1, auc))
        else:
            convnext.append((ep, wll, ba, f1, auc))
    except:
        pass

print('=== SPINEXNET v3 ===')
print('Ep  Concept   WLL     BalAcc  F1      AUC')
best_s = min(r[2] for r in spinexnet)
for ep, c, w, b, f, a in spinexnet:
    tag = ' *BEST*' if abs(w - best_s) < 0.0001 else ''
    print(f'{ep:2d}  {c:.4f}  {w:.4f}  {b:.4f}  {f:.4f}  {a:.4f}{tag}')

print()
print('=== CONVNEXT BASELINE ===')
print('Ep  WLL     BalAcc  F1      AUC')
best_c = min(r[1] for r in convnext)
for ep, w, b, f, a in convnext:
    tag = ' *BEST*' if abs(w - best_c) < 0.0001 else ''
    print(f'{ep:2d}  {w:.4f}  {b:.4f}  {f:.4f}  {a:.4f}{tag}')
