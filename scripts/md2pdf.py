#!/usr/bin/env python3
"""Minimal Markdown -> LaTeX converter for the write-ups in paper/.

There is no pandoc on this machine, so this covers just the subset those
documents use: ATX headings, pipe tables, bold, italic, inline code, bullet
lists, horizontal rules, blockquotes, and the handful of non-ASCII characters
that appear (section sign, degree, times, dashes, arrow, minus).

Emits a .tex file; run pdflatex yourself (twice, for the table of contents
and hyperref outlines):

    python scripts/md2pdf.py paper/inversion_story.md /tmp/x/story.tex
    cd /tmp/x && pdflatex -interaction=nonstopmode story.tex && pdflatex ...
    cp story.pdf paper/inversion_story.pdf

Wide tables are set in tabularx with ragged-right wrapped columns; if a table
still overflows, lower BUDGET in table().
"""
import re
import sys

UNI = {
    '\u00a7': r'\S{}', '\u00b0': r'$^\circ$', '\u00d7': r'$\times$',
    '\u2013': '--', '\u2014': '---', '\u2192': r'$\rightarrow$',
    '\u2212': '$-$',
}
SPECIALS = {'&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_',
            '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}',
            '^': r'\textasciicircum{}'}


def esc(s):
    s = s.replace('\\', r'\textbackslash{}')
    s = ''.join(SPECIALS.get(c, c) for c in s)
    for k, v in UNI.items():
        s = s.replace(k, v)
    return s


def inline(s):
    """Escape, then apply inline markdown. Code spans are protected first."""
    codes = []

    def stash(m):
        codes.append(m.group(1))
        return f'\x00{len(codes) - 1}\x00'

    s = re.sub(r'`([^`]+)`', stash, s)
    s = esc(s)
    s = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', s)
    s = re.sub(r'(?<!\*)\*([^*]+?)\*(?!\*)', r'\\emph{\1}', s)
    s = re.sub(r'\[(.+?)\]\((.+?)\)', r'\1', s)          # links -> text
    return re.sub(r'\x00(\d+)\x00',
                  lambda m: r'\texttt{' + esc(codes[int(m.group(1))]) + '}', s)


def split_row(line):
    return [c.strip() for c in line.strip().strip('|').split('|')]


def table(rows):
    head, body = rows[0], rows[2:]
    ncol = len(head)
    widest = [max([len(r[i]) for r in [head] + body if i < len(r)] or [0])
              for i in range(ncol)]
    wrap = [w > 32 for w in widest]
    # A row can also overflow from many narrow columns; at \footnotesize the
    # line holds roughly 85 characters. Wrap the widest columns until the
    # unwrapped ones fit.
    BUDGET = 85
    while sum(w for w, k in zip(widest, wrap) if not k) > BUDGET:
        cand = [(w, i) for i, (w, k) in enumerate(zip(widest, wrap)) if not k]
        if not cand or max(cand)[0] <= 6:
            break
        wrap[max(cand)[1]] = True
    if any(wrap):
        spec = ''.join('Y' if wrap[i] else 'l' for i in range(ncol))
        env, arg = 'tabularx', r'{\linewidth}{@{}' + spec + '@{}}'
    else:
        env, arg = 'tabular', '{@{}' + 'l' * ncol + '@{}}'
    out = [r'\begin{center}', r'\footnotesize', f'\\begin{{{env}}}{arg}',
           r'\toprule',
           ' & '.join(r'\textbf{%s}' % inline(c) for c in head) + r' \\',
           r'\midrule']
    for r in body:
        r = (r + [''] * ncol)[:ncol]
        out.append(' & '.join(inline(c) for c in r) + r' \\')
    out += [r'\bottomrule', f'\\end{{{env}}}', r'\end{center}']
    return out


def convert(md):
    lines = md.split('\n')
    out, i, in_list = [], 0, False

    def close_list():
        nonlocal in_list
        if in_list:
            out.append(r'\end{itemize}')
            in_list = False

    while i < len(lines):
        ln = lines[i].rstrip()

        if ln.startswith('|'):
            close_list()
            block = []
            while i < len(lines) and lines[i].lstrip().startswith('|'):
                block.append(split_row(lines[i]))
                i += 1
            if len(block) >= 2:
                out += table(block)
            continue

        if re.fullmatch(r'-{3,}', ln.strip()):
            close_list()
            out.append(r'\vspace{2ex}\hrule\vspace{2ex}')
        elif ln.startswith('#'):
            close_list()
            lvl = len(ln) - len(ln.lstrip('#'))
            cmd = {1: 'section', 2: 'subsection'}.get(lvl, 'subsubsection')
            out.append(f'\\{cmd}*{{{inline(ln.lstrip("#").strip())}}}')
        elif ln.startswith('> '):
            close_list()
            out.append(r'\begin{quote}' + inline(ln[2:]) + r'\end{quote}')
        elif re.match(r'^\s*[-*]\s+', ln):
            if not in_list:
                out.append(r'\begin{itemize}')
                in_list = True
            item = [re.sub(r'^\s*[-*]\s+', '', ln)]
            while (i + 1 < len(lines) and lines[i + 1].startswith('  ')
                   and lines[i + 1].strip()
                   and not re.match(r'^\s*[-*]\s+', lines[i + 1])):
                i += 1
                item.append(lines[i].strip())
            out.append(r'\item ' + inline(' '.join(item)))
        elif not ln.strip():
            close_list()
            out.append('')
        else:
            para = [ln]
            while (i + 1 < len(lines) and lines[i + 1].strip()
                   and not lines[i + 1].lstrip().startswith(('|', '#', '>', '-'))
                   and not re.match(r'^\s*[-*]\s+', lines[i + 1])):
                i += 1
                para.append(lines[i].strip())
            close_list()
            out.append(inline(' '.join(para)))
        i += 1
    close_list()
    return '\n'.join(out)


PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[a4paper,margin=2.2cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{booktabs,tabularx,array,amssymb,microtype}
% Wrapped table columns: ragged right, and never hyphenate a word such as
% VGGFace2 across two lines of a narrow cell.
\newcolumntype{Y}{>{\raggedright\arraybackslash\hyphenpenalty=10000%
\exhyphenpenalty=10000\relax}X}
\usepackage[colorlinks=true,linkcolor=black,urlcolor=black]{hyperref}
\usepackage{parskip}
\renewcommand{\arraystretch}{1.15}
\setcounter{secnumdepth}{0}
\title{@TITLE@}
\date{}
\begin{document}
\maketitle
\vspace{-3em}
"""

if __name__ == '__main__':
    src, dst = sys.argv[1], sys.argv[2]
    md = open(src, encoding='utf-8').read()
    m = re.match(r'#\s+(.+)', md)
    title = inline(m.group(1)) if m else 'Untitled'
    if m:
        md = md[m.end():]
    body = convert(md)
    open(dst, 'w', encoding='utf-8').write(
        PREAMBLE.replace('@TITLE@', title) + body + '\n\\end{document}\n')
    print(f'wrote {dst}')
