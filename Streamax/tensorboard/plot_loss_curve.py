# yangbing
import json
import os
from tensorboardX import SummaryWriter

root = './'
json_name = 'metrics.json'

writer = SummaryWriter(os.path.join(root, 'logs'))
file = open(os.path.join(root, json_name), 'r', encoding='utf-8')

batch = 0
for line in file.readlines():
    d = json.loads(line)
    writer.add_scalar('3.2_4/lr', d['lr'], batch)
    writer.add_scalar('3.2_4/loss_fcos_cls', d['loss_cls'], batch)
    writer.add_scalar('3.2_4/loss_fcos_ctr', d['loss_bbox'], batch)
    writer.add_scalar('3.2_4/loss_fcos_loc', d['loss_mask'], batch)
    writer.add_scalar('3.2_4/total_loss', d['loss_centerness'], batch)
    batch += 1
writer.close()
