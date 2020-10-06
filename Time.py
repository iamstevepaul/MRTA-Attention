import pickle
file1 = open('outputs/Results_200_Att_Latest/output_mrta_att.out', 'r')
Lines = file1.readlines()

count = 0
# Strips the newline character
i=0
att_time = []
for line in Lines:
    if line[0:7] == "Epoch: ":
        time = float(line[19:30])
        if i < 200:
            att_time.append(time)
        i +=1

file2 = open('outputs/Results_200_ccn_new_2/output_mrta.out', 'r')
Lines = file2.readlines()

count = 0
# Strips the newline character
i=0
ccn_time = []
for line in Lines:
    if line[0:7] == "Epoch: ":
        time = float(line[19:30])
        if i < 200:
            ccn_time.append(time)
        i +=1
Data = {
    "CAM": ccn_time,
    "AM": att_time
}
# with open('Epoch_time.pkl', 'wb') as f:
#     pickle.dump(Data, f, pickle.HIGHEST_PROTOCOL)
ATT_Time = [12.75,
            11.76,
            13.95,
            15.98,
            16.812,
            17.023,
            16.83,
            16.66,
            16.75,
            15.48,
            14.31,
            11.32,
            11.18,
            11.05,
            11.05,
            11.30,
            11.20,
            12.63,
            11.62,
            11.95,
            11.63,
            13.85,
            13.68,
            15.16,
            14.49,
            16.25,
            14.88,
            15.46,
            15.60,
            16.39,
            17.00,
            15.94,
            15.12,
            15.44,
            16.00,
            15.26,
            15.65,
            15.46,
            14.82,
            15.04,
            15.94,
            16.10,
            15.78,
            16.02,
            16.04,
            16.00,
            16.13,
            16.32,
            16.23,
            16.72,
            16.58,
            15.97
            ]