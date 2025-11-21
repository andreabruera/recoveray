import os
import pdf2image
import pylatex

from pdf2image import convert_from_path
from pylatex import Document, Section, Subsection, Command
from pylatex.utils import italic, NoEscape

os.system("find prediction_plots -name '*pdf' -exec rm {} \;")

latex_document = 'path'
#start = '\documentclass{article}\n\\begin{document}\n\\begin{table}[h!]\n\centering\n'
#start = '\\begin{table}[h!]\n\centering\n'
start = '\\hspace*{-3.cm}\\begin{table}[h!]\n'
#end =  '\hline\n\end{tabular}\n\end{table}\n\end{document}'
end =  '\hline\n\end{tabular}\n\end{table}'
'''
 \begin{tabular}{||c c c c||}
 \hline
 Col1 & Col2 & Col2 & Col3 \\ [0.5ex]
 \hline\hline
 1 & 6 & 87837 & 787 \\
 2 & 7 & 78 & 5415 \\
 3 & 545 & 778 & 7507 \\
 4 & 545 & 18744 & 7560 \\
 5 & 88 & 788 & 6344 \\ [1ex]
'''

for root, direc, fz in os.walk('prediction_plots'):
    for f in fz:
        ### for each document we generate a table
        if 'tsv' not in f:
            continue
        lines = list()
        with open(os.path.join(root, f)) as i:
            for l in i:
                line = l.strip().split('\t')
                lines.append(line)
        n_cols = set([len(l) for l in lines])
        assert len(n_cols) == 1
        n_cols = list(n_cols)[0]
        latex_lines = [start,]
        for l_i, l in enumerate(lines):
            ### headers
            if l_i == 0:
                header = '\\begin{tabular}{||'
                for _ in range(n_cols):
                    header += 'c'
                    if _ < n_cols-1:
                        header += ' '
                header += '||}\n\hline\n'
                subheader = ''
                for _, h in enumerate(l):
                    l = h.replace('_', ' ').replace('%', '\%').replace('spe', 'Spe').replace('rho', '$\\rho$')
                    subheader += l[0].upper()+l[1:].replace(' on Spearman', ' on ')
                    if _ < n_cols-1:
                        subheader += ' & '
                    else:
                        subheader += '\\\\ [0.5ex]\n\hline\hline\n'
                latex_lines.append(header)
                latex_lines.append(subheader)
            ### other lines
            else:
                subline = ''
                for _, h in enumerate(l):
                    ### check significance
                    if float(l[-2].replace(',', '.')) < 0.05:
                        if 'mpact' in lines[0][-4]:
                            if float(l[-4].replace(',', '.')) < 0.:
                                sig = True
                            else:
                                sig = False
                        else:
                            sig = True
                    else:
                        sig = False
                    try:
                        if '(' not in h:
                            if 'rho' in lines[0][_]:
                                lim = 3
                            else:
                                lim = 4
                            if sig and _ != len(l)-1:
                                subline += '\\textbf{'+str(round(float(h.replace(',',  '.')), lim))+'}'
                            else:
                                subline += str(round(float(h.replace(',',  '.')), lim))
                        else:
                            first = round(float(h[1:-1].split(',')[0]), 3)
                            second = round(float(h[1:-1].split(',')[1]), 3)
                            subline += '( {}, {} )'.format(first, second)
                    #except (TypeError, ValueError):
                    except ValueError:
                        new_l = h[0].upper()+h[1:].replace('_', ' ').replace(' all', '').replace('lSMA', 'SMA').replace('L SMA',  'SMA').replace('w/o l', 'w/o L ').replace('w/o r', 'w/o R ').replace('to l', 'to L ').replace('to r', 'to R ').replace('mprovements',  'mprov.').replace('bilities', 'bility').replace('ivations', 'ivity').replace('esions', 'esion')
                        if 'w/o' in new_l and 'esion' not in new_l:
                            new_l = new_l[:-3]
                        if len(new_l) == 2:
                            new_l = 'Ability {}'.format(new_l)
                        if sig:
                            subline += '\\textbf{'+new_l.strip()+'}'
                        else:
                            subline += new_l.strip()
                    if _ < n_cols-1:
                        subline += ' & '
                    else:
                        subline += '\\\\\n'
                latex_lines.append(subline)
        latex_lines.append(end)
        tex = ''.join(latex_lines)

        doc = Document('basic', indent=False,geometry_options={'left' : '20mm', 'top' : '0mm', 'right' : '0mm'}, font_size='Huge')
        doc.append(NoEscape(tex))
        out_f = os.path.join(root, f).replace('.tsv', '')
        doc.generate_pdf(out_f, clean_tex=False)
        imgs = convert_from_path(out_f+'.pdf', fmt="jpeg", output_file=out_f+'_table.jpg')
        assert len(imgs) == 1
        imgs[0].save(out_f+'_table.jpg')
os.system("find prediction_plots -name '*tex' -exec rm {} \;")
