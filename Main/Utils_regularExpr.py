import subprocess
from xeger import Xeger


def getRegularExpressionForStrings(ListOfStrings):
    arguments = ' '.join(ListOfStrings)

    # Returning an empty string if empty was given
    if len(arguments.strip())== 0:
        return " "

    cmd = "regexgen " + arguments
    #print("command is " + cmd)
    output = subprocess.check_output(cmd, shell=True)
    output = output.decode('utf-8')
    output = output[1:]
    output = output[:-2]
    output = output.replace('?:', '')
    return output


def infereRegularExpression(regExpr, limit=10):
    x = Xeger(limit=10)
    return x.xeger(regExpr)


def test():
    regExpr = getRegularExpressionForStrings(["foo", "foobar", "foobiz", "paduraru"])

    for i in range(1000):
        print(infereRegularExpression(regExpr))
