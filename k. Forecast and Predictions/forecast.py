import exrex

def mexico_national_id():
    sum = 0
    weights = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
               'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'I': 18, 'J': 19,
               'K': 20, 'L': 21, 'M': 22, 'Ñ': 23, 'N': 24, 'O': 25, 'P': 26, 'Q': 27, 'R': 28, 'S': 29,
               'T': 30, 'U': 31, 'V': 32, 'W': 33, 'X': 34, 'Y': 35, 'Z': 36}

    # generate regex
    nat_id = exrex.getone("([A-ZÑ]{4}[\d]{6}[HM](AG|AS|BC|BS|CM|CC|CS|CH|CO|CL|DF|DG|GT|GR|HG|JA|JC|EM|MI|MO|MS|MC|MN|NA|NT|NL|OA|OC|PU|QT|QR|SL|SI|SO|SP|SR|TB|TC|TS|TM|TL|VE|YU|ZA|NE|VZ|YN|ZS)[A-Z]{3}[A-Z\dÑ]\d)")
    print("Mexico National ID:"+nat_id)

    # for every digit in id
    for i in range(0, 18):

        # sum up weights from dict
        sum += weights.get(nat_id[i]) * (18 - i)

    # if modulo 10 equals 0, this passes
    if sum % 10 == 0:
        print('pass')

def spain_national_id():
    alphabet = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7, 'H':8, 'I':9, 'J':10,
                'K':11, 'L':12, 'M':13, 'N':14, 'O':15, 'P':16, 'Q':17, 'R':18, 'S':19,
                'T':20, 'U':21, 'V':22, 'W':23, 'X':24, 'Y':25, 'Z':26}

    reminderLetters = ['T', 'R', 'W', 'A', 'G',
                       'M', 'Y', 'F', 'P', 'D', 'X', 'B', 'N', 'J',
                       'Z', 'S', 'Q', 'V', 'H', 'L', 'C', 'K', 'E']

    # generate regex
    nat_id = exrex.getone("\d{8}-[A-Z]")
    print("Spain National ID:"+nat_id)

    # convert char to number
    numerical_nat_id = str(nat_id[0:8]) + str(alphabet[nat_id[9]])

    # generate reminder
    reminder = int(numerical_nat_id) % 23
    controlChar = nat_id[9]

    # if control char equals reminder letters, accept
    if controlChar == reminderLetters[reminder]:
        print('pass')

def mexico_tax_id():

    sum = 0
    weights = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
               'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'I': 18, 'J': 19,
               'K': 20, 'L': 21, 'M': 22, 'N': 23, '&':24, 'O': 25, 'P': 26, 'Q': 27, 'R': 28, 'S': 29,
               'T': 30, 'U': 31, 'V': 32, 'W': 33, 'X': 34, 'Y': 35, 'Z': 36, ' ':37, 'Ñ': 38}

    # generate regex
    tax_id = exrex.getone("([A-ZÑ]{4}[\d]{6}[0-9A-ZÑ]{3})")
    print("Mexico Tax ID:"+tax_id)

    # for every digit in id
    for i in range(0, 13):

        # sum up weights from dict
        sum += weights.get(tax_id[i]) * (13 - i)

    # if modulo 10 equals 0, this passes
    if sum % 11 == 0:
        print('pass')

def spain_tax_id():

    alphabet = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7, 'H':8, 'I':9, 'J':10,
                'K':11, 'L':12, 'M':13, 'N':14, 'O':15, 'P':16, 'Q':17, 'R':18, 'S':19,
                'T':20, 'U':21, 'V':22, 'W':23, 'X':24, 'Y':25, 'Z':26}

    CHECK_LETTER_TABLE = "TRWAGMYFPDXBNJZSQVHLCKE"

    # generate regex
    tax_id = exrex.getone("(\d{8}[A-Z])|([XYZKLM]\d{7}[A-Z])")
    print("Spain Tax ID:"+tax_id)

    # generate number version
    numerical_tax_id = ""

    # convert char to number
    for i in tax_id:
        if i.isnumeric():
            numerical_tax_id = numerical_tax_id + str(i)
        else:
            numerical_tax_id = numerical_tax_id + str(alphabet[i])

    # first check
    reminder = int(numerical_tax_id) % 23
    if tax_id[8] == CHECK_LETTER_TABLE[reminder]:
        print('pass')
        return

    # second check
    else:
        ch = tax_id[0]
        if ch == 'X' or ch == 'K' or ch == 'L' or ch == 'M':
            numerical_tax_id = '0' + str(numerical_tax_id)
        elif ch == 'Y':
            numerical_tax_id = '1' + str(numerical_tax_id)
        elif ch == 'Z':
            numerical_tax_id = '2' + str(numerical_tax_id)
        else:
            return

        reminder = int(numerical_tax_id) % 23

        if tax_id[8] == CHECK_LETTER_TABLE[reminder]:
            print('pass')

def port_routing_number():
    sum = 0
    weights = [3, 7, 1, 3, 7, 1, 3, 7, 1]

    # generate regex
    rout_num = exrex.getone("\d{9}")
    print("Port Routing Number:"+rout_num)

    for i in range(0,9):
        sum += ord(rout_num[i]) * weights[i]

    if sum % 10 == 0:
        print('pass')

def port_acc_num():

    WEIGHTS = [2, 7, 6, 5, 4, 3, 2]

    # generate regex
    acc_num_1 = exrex.getone("(\d{7}[- ]?|\d{4}\\.\d{3}\\-)[\d]")
    acc_num_2 = exrex.getone("(\d{8})")
    acc_num_3 = exrex.getone("(\d{7}[- ]?|\d{4}\\.\d{3}\\-)[P]")
    print("Port Acc Number:"+ acc_num_3)

    acc_num = acc_num_3

    total = 0
    if acc_num[4] == '.':
        k = 0
        for i in range(0,8):
            if acc_num[i] == '.' or acc_num[i] == '-':
                continue
            num = int(acc_num[i])
            if num > 9 or num < 0 or str(num) == None:
                return
            else:
                total += num * WEIGHTS[k]
                ++k
    else:
        for i in range(0,7):
            num = int(acc_num[i])
            if num > 9 or num < 0 or str(num) == None:
                return
            else:
                total += num * WEIGHTS[i]

    ret = 11 - (total % 11)
    answer = False
    last = acc_num[-1:]
    if (last <= '9' and last >= '0') or (last == 'P'):
        if ret < 10:
            answer = (int(ord(last)) == ret)
        else:
            if ret == 10:
                answer = (str(last) == 'P')
            else:
                answer = (str(last) == '0')
    print(answer)

def port_bank_branch():

    weights = [5,4,3,2]
    total = 0

    # generate regex
    bank_branch = exrex.getone("\d{4}(\\-\d)")
    print("Port Bank Branch:"+bank_branch)

    for i in range(0,4):
        value = int(bank_branch[i])
        if value > 9 or value < 0:
            return

        total += value * weights[i]

    ret = 11 - (total % 11)
    last = str(bank_branch)[-1:]
    if int(last) <= 9 and int(last) >= 0:
        if ret < 10:
            answer = (int(last) == ret)
        else:
            answer = (last == 0)
    print(answer)

def port_health_id():

    # generate regex
    health_id = exrex.getone("([12]\d{10}00[01]\d)|([789]\d{14})")
    print("Port Health ID:"+health_id)

    sum = 0
    for i in range (0,15):
        sum += int(health_id[i]) * (15 - i)
    if sum % 11 == 0:
        print('pass')

def port_national_id():

    # generate regex
    nat_id_1 = exrex.getone("\d{8}-[\dx]")
    nat_id_2 = exrex.getone("(\d{10}\\-?\d)")
    print("Port National ID:" + nat_id_2)

    nat_id = nat_id_2

    checkSeq = [2, 3, 4, 5, 6, 7, 8, 9]
    sum = 0
    for i in range(0,8):
        sum += int(nat_id[i]) * checkSeq[i]

    checkDigit = 0
    nat_id_clear = nat_id.replace('-','')
    last = str(nat_id_clear)[8]
    if last == 'x':
        checkDigit = 10
    elif last == 0:
        checkDigit = 11
    else:
        checkDigit = last

    if (11 - (sum % 11) == checkDigit):
        print('pass')

def port_employee_id():

    weights = [3, 2, 9, 8, 7, 6, 5, 4, 3, 2]

    # generate regex
    emp_id_1 = exrex.getone("\d{3}\\.\d{4}\\.\d{3}-\d|\d{3}\\.\d{5}\\.\d{2}-\d")
    emp_id_2 = exrex.getone("\d{11}")
    print("Port Employee ID:" + emp_id_2)

    emp_id = emp_id_2
    emp_id = emp_id.replace('.',"").replace('-','')
    total = 0
    for i in range(0,10):
        total += int(emp_id[i]) * weights[i]

    ret = 11 - (total % 11)
    if ret == 11 or ret == 10:
        ret = 0

    if ret == int(emp_id[10]):
        print('pass')

def port_tax_id():

    cpf_size = 11

    # generate regex
    tax_id_1 = exrex.getone("[\d]{11}")
    tax_id_2 = exrex.getone("([\d]{3}\\.[\d]{3}\\.[\d]{3}-[\d]{2})")
    tax_id_3 = exrex.getone("([\d]{3}-[\d]{3}-[\d]{3}-?[\d]{2})")
    print("Port Tax ID:" + tax_id_1)

    tax_id = tax_id_1
    tax_id = tax_id.replace('.','').replace('-','')


if __name__ == '__main__':
    for i in range(100):
        port_tax_id()