import socket
hostname = socket.gethostname()

if hostname == 'crunchy.cl.cam.ac.uk':
    data_path = '/local/sdb/wd263/PGM/neutral/'
    img_save_path = '/local/sdb/wd263/PGM/image_gen_sm/'
    model_save_path = '/local/sdb/wd263/PGM/'
elif hostname == 'kiiara.cl.cam.ac.uk':
    data_path = '/local/scratch/wd263/PGM/neutral/'
    img_save_path = '/local/scratch/wd263/PGM/image_gen_sm/'
    model_save_path = '/local/scratch/wd263/PGM/'
else:
    data_path = '/rds-d2/project/t2_vol2/rds-t2-cs056/wd263/PGM/neutral/'
    data_m_path = '/rds-d3/user/wd263/hpc-work/PGM/neutral_merged_m/' 
    data_b_path = '/rds-d3/user/wd263/hpc-work/PGM/neutral_merged_b/'
    data_8s_path = '/rds-d3/user/wd263/hpc-work/PGM/neutral_merged_8s/'
    data_bs_path = '/rds-d2/project/t2_vol2/rds-t2-cs056/wd263/PGM/neutral_merged_bs/'
    img_save_path = '/rds-d3/user/wd263/hpc-work/PGM/image_gen/'
    model_save_path = '/rds-d3/user/wd263/hpc-work/PGM/'








