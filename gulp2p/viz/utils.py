
def myround(x, prec=0, base=360):
  # https://stackoverflow.com/questions/2272149/round-to-5-or-other-number-in-python
  return round(base * round(float(x)/base),prec)
