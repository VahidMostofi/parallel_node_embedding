import sys
from subprocess import PIPE, Popen
from threading  import Thread
from queue import Queue, Empty
import os

ON_POSIX = 'posix' in sys.builtin_module_names

def generate_node2vec_command(input_file_path, output_file_path):
    node2vec_executer_path = '/Users/vahid/Desktop/projects/snap/examples/node2vec/node2vec'
    command = ['time']
    command.append(node2vec_executer_path)
    command.append('-i:' + input_file_path)
    command.append('-o:' + output_file_path)
    return command

def enqueue_output(out, queue, task_name):
    for line in iter(out.readline, b''):
        line = str(line)
        queue.put(task_name + '::' + line)
    out.close()

def run_task(task_name, working_dir, verbose=False):
    if verbose:
        print('running task {}'.format(task_name))

    input_file_path  = working_dir + task_name + '_edges.txt'
    output_file_path = working_dir + 'embeddings/' + task_name + '.emb'
    command = generate_node2vec_command(input_file_path,  output_file_path)
    p = Popen(command, stdout=PIPE, stderr=PIPE, bufsize=1, close_fds=ON_POSIX)
    t = Thread(target=enqueue_output, args=(p.stderr, q, task_name))
    t.daemon = True
    t.start()

def run_node2vec_combinations(combinations,working_dir, concurrent_count = 6, verbose=False):
    global q
    q = Queue()
    embeddings_folder = working_dir + 'embeddings/'
    print('running {} tasks for {} of combinations.'.format(len(combinations), len(combinations)))
    running_times = []

    if not os.path.exists(embeddings_folder):
        os.makedirs(embeddings_folder)

    all_tasks = set()
    running_tasks = set()
    done_tasks = set()

    for comb in combinations:
        p1 = int(comb[0])
        p2 = int(comb[1])
        all_tasks.add(str(min(p1,p2)) + '_' + str(max(p1,p2)))
    
    while True:
        
        while len(running_tasks) < concurrent_count and  len(all_tasks) > 0:
            new_task = all_tasks.pop()
            run_task(new_task, working_dir, verbose=verbose)
            running_tasks.add(new_task)

        try:  line = q.get(timeout=1)
        except Empty:
            pass
        else:
            task_name, message = line.split('::')
            
            if 'real' in message and 'user' in message and  'sys' in message:
                message = str(message)
                message = message.replace("b'",'')
                time_took = message.split(' real')[0].strip()
                print('task {} took {}. {} tasks left'.format(task_name, time_took, len(all_tasks)+len(running_tasks)-1))
                running_times.append((task_name, time_took))
                done_tasks.add(task_name)
                running_tasks.remove(task_name)
        if len(all_tasks) == 0 and len(running_tasks) == 0:
            break
    return running_times

