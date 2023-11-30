##Power Analysis of Deming 208CRC Ctrl vs Folfox Data
##Power analysis, t-test

library(pwr)

##208CRC
M1  =  0.01908                     # Mean for sample 1 (Control)
M2  = -0.06661                     # Mean for sample 2 (FOLFOX)
S1  =  0.033898                    # Std dev for sample 1
S2  =  0.042588                    # Std dev for sample 2

Cohen.d1 = (M1 - M2)/sqrt(((S1^2) + (S2^2))/2) 
Glass.d1 = (M1 - M2)/(S1)

pwr.t.test(
  n = NULL,                  # Observations in _each_ group
  d = Glass.d1,           
  sig.level = 0.05,          # Type I probability
  power = 0.8,              # 1 minus Type II probability
  type = "two.sample",       # Change for one- or two-sample
  alternative = "two.sided"
)

##194CRC
M3  = -0.01955                    # Mean for sample 1 (control)
M4  = -0.06303                    # Mean for sample 2 (FOLFOX)
S3  =  0.03262                    # Std dev for sample 1
S4  =  0.05643                    # Std dev for sample 2

Cohen.d2 = (M3 - M4)/sqrt(((S3^2) + (S4^2))/2) 
Glass.d2 = (M3 - M4)/(S3)

pwr.t.test(
  n = NULL,                  # Observations in _each_ group
  d = Glass.d2,           
  sig.level = 0.05,          # Type I probability
  power = 0.8,              # 1 minus Type II probability
  type = "two.sample",       # Change for one- or two-sample
  alternative = "two.sided"
)
