from datetime import date

TODAY = date.today().strftime("%Y-%m-%d")
test = "2022-05-04"

print(TODAY < test)