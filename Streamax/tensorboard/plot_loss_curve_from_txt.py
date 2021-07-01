# sunxin
import json
import os
from tensorboardX import SummaryWriter

root = './'
txt_name = 'log.txt'

writer = SummaryWriter(os.path.join(root, 'logs'))
file = open(os.path.join(root, txt_name), 'r', encoding='utf-8')



iter_batch = 0
for line in file.readlines():
    if '  iter:' in line:
        dict_this_line = {}
        items = line.split('  ')
        for item in items:
            if 'INFO' not in item:
                item = str(item)
                item = item.replace(' ', '')
                item = item.replace('\n', '')
                if '(' in item:
                    idx = item.find('(')
                    item = item[:idx]
                key = item.split(':')[0]
                val = str(item.split(':')[1]).rstrip('(')
                if 'nan' in val:
                    val = 100
                    # pass
                else:
                    if '.' in val:
                        val = float(val)
                    else:
                        val = int(val)
                dict_this_line.setdefault(key, val)
        print(dict_this_line)






        writer.add_scalar('baseline/lr', dict_this_line['lr'], iter_batch)
        writer.add_scalar('baseline/loss_centerness', dict_this_line['loss_centerness'], iter_batch)
        writer.add_scalar('baseline/loss_cls', dict_this_line['loss_cls'], iter_batch)
        writer.add_scalar('baseline/loss_reg', dict_this_line['loss_reg'], iter_batch)
        writer.add_scalar('baseline/loss', dict_this_line['loss'], iter_batch)
        iter_batch += 1
writer.close()



