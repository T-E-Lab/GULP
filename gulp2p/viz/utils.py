

def myround(x, prec=2, base=.05):
  # https://stackoverflow.com/questions/2272149/round-to-5-or-other-number-in-python
  return round(base * round(float(x)/base),prec)
