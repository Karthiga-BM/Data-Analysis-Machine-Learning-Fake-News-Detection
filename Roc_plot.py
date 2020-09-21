

from matplotlib import pyplot as plt
#Accuracy Comparison for all four models with 5 fold cross validation and BOW
plt.figure()
plt.title("5-fold cross validation + BOW_Models Accuracy Comparison")
plt.plot([100, 500, 1000, 5000, 8000], [0.633542977, 0.68427673, 0.726065688, 0.783647799, 0.79301188], label="Logisic Regression Model", color = 'purple')
plt.plot([100, 500, 1000, 5000, 8000], [0.626974144, 0.681761006, 0.717260657, 0.778057303, 0.787002096], label="Naive Bayes")
plt.plot([100, 500, 1000, 5000, 8000], [0.631027254, 0.668204053, 0.695317959, 0.708315863, 0.709014675], label="Random Forest")
plt.plot([100, 500, 1000, 5000, 8000], [0.624039133,0.67966457, 0.719357093, 0.767714885,0.776799441], label="SVM")
plt.xlabel("Number of Features")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig('5-fold cross validation + BOW_Models Accuracy Comparison')
plt.show()
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Accuracy Comparison for all four models with 5 fold cross validation and Tf-idf
plt.figure()
plt.title("5-fold cross validation + Tf-idf_Models Accuracy Comparison")
plt.plot([100, 500, 1000, 5000, 8000], [0.736687631, 0.775960867, 0.802655486, 0.830328442, 0.836477987], label="Logisic Regression Model")
plt.plot([100, 500, 1000, 5000, 8000], [0.721872816, 0.762264151, 0.782250175,0.827113906, 0.833682739], label="Naive Bayes")
plt.plot([100, 500, 1000, 5000, 8000], [0.735569532, 0.765059399, 0.777777778,0.789797345, 0.792173305], label="Random Forest")
plt.plot([100, 500, 1000, 5000, 8000], [0.734730957,0.780852551, 0.801537386,0.827952481,0.833403215], label="SVM")
plt.xlabel("Number of Features")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig('5-fold cross validation + Tf_idf_Models Accuracy Comparison')
plt.show()
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Mean_Square Error Comparison for all four models with 5 fold cross validation and BOW
plt.figure()
plt.title("5-fold cross validation + BOW_Models Mean Square Error Comparison")
plt.plot([100, 500, 1000, 5000, 8000], [0.605356939,0.561892579, 0.523387344, 0.465136755, 0.454959471], label="Logisic Regression Model")
plt.plot([100, 500, 1000, 5000, 8000], [0.610758427,0.564126753, 0.531732398, 0.471107947, 0.461516959], label="Naive Bayes")
plt.plot([100, 500, 1000, 5000, 8000], [0.607431269, 0.576017315, 0.551980109, 0.540077899, 0.539430556], label="Random Forest")
plt.plot([100, 500, 1000, 5000, 8000], [0.613156478,0.565981828, 0.529757404,0.481959661,0.472441064], label="SVM")
plt.xlabel("Number of Features")
plt.ylabel("Mean Squared error")
plt.legend()
plt.savefig('5-fold cross validation + BOW_Models Mean Square Error Comparison')
plt.show()
#---------------------------------------------------------------------------------------------------------------------------------------------------------------

#Mean Square Error Comparison for all four models with 5 fold cross validation and Tf-idf
plt.figure()
plt.title("5-fold cross validation + Tf-idf_Models Mean Square Error Comparison")
plt.plot([100, 500, 1000, 5000, 8000], [0.513139717,0.473327723, 0.444234751, 0.411912076, 0.404378551], label="Logisic Regression Model")
plt.plot([100, 500, 1000, 5000, 8000], [0.527377648,0.487581633, 0.466636717, 0.415795735, 0.407820133], label="Naive Bayes")
plt.plot([100, 500, 1000, 5000, 8000], [0.514228031, 0.484706716, 0.471404521, 0.458478631,0.455880132], label="Random Forest")
plt.plot([100, 500, 1000, 5000, 8000], [0.515042758,0.468131872, 0.445491429,0.414786113,0.408162695], label="SVM")
plt.xlabel("Number of Features")
plt.ylabel("Mean Squared error")
plt.legend()
plt.savefig('5-fold cross validation + Tf-idf_Models Mean Square Error Comparison')
plt.show()


#---------------------------------------------------------------------------------------------------------------------------------------------------------------

#F1 Score Comparison for LR
plt.figure()
plt.title("F1 score comparison of Logistic Regression  for BOW and Tf-idf")
plt.plot([100, 500, 1000, 5000, 8000], [0.74, 0.78,0.80,0.83,0.84], label="Logisic Regression Model + Tf-idf")
plt.plot([100, 500, 1000, 5000, 8000], [0.62, 0.68, 0.72, 0.78, 0.79], label="Logisic Regression Model + BOW")
plt.xlabel("Number of Features")
plt.ylabel("F1 Score")
plt.legend()
plt.savefig('F1 score comparison of Logistic Regression  for BOW and Tf-idf')
plt.show()
#---------------------------------------------------------------------------------------------------------------------------------------------------------------

#F1 Score Comparison for NB
plt.figure()
plt.title("F1 score comparison of Naive Bayes for BOW and Tf-idf")
plt.plot([100, 500, 1000, 5000, 8000], [0.72, 0.76,0.78,0.83,0.83], label="Naive Bayes + Tf-idf", color = 'purple')
plt.plot([100, 500, 1000, 5000, 8000], [0.61, 0.68, 0.72, 0.78, 0.79], label="Naive Bayes + BOW", color = 'deepskyblue')
plt.xlabel("Number of Features")
plt.ylabel("F1 Score")
plt.legend()
plt.savefig('F1 Score  Comparison of Naive Bayes Model for BOW and Tf-idf')
plt.show()
#---------------------------------------------------------------------------------------------------------------------------------------------------------------

#F1 Score Comparison for RF
plt.figure()
plt.title("F1 score comparison of Random Forest  for BOW and Tf-idf")
plt.plot([100, 500, 1000, 5000, 8000], [0.74, 0.76,0.78,0.79,0.79], label="Random Forest + Tf-idf", color = 'red')
plt.plot([100, 500, 1000, 5000, 8000], [0.61, 0.65, 0.68, 0.69, 0.70], label="Random Forest + BOW", color = 'darkblue')
plt.xlabel("Number of Features")
plt.ylabel("F1 Score")
plt.legend()
plt.savefig('F1 score comparison of Random Forest for BOW and Tf-idf')
plt.show()
#---------------------------------------------------------------------------------------------------------------------------------------------------------------

#F1 Score Comparison for SVM


plt.figure()
plt.title("F1 score comparison of SVM  for BOW and Tf-idf")
plt.plot([100, 500, 1000, 5000, 8000], [0.73, 0.78,0.80,0.83,0.83], label="SVM + Tf-idf", color ='teal')
plt.plot([100, 500, 1000, 5000, 8000], [0.60, 0.67, 0.72, 0.77, 0.78], label="SVM+ BOW", color ='magenta')
plt.xlabel("Number of Features")
plt.ylabel("F1 Score")
plt.legend()
plt.savefig('F1 score comparison of SVM  for BOW and Tf-idf')
plt.show()

#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#Comparing SVM and Naive Bayes:
plt.figure()
plt.title("Comapring SVM and Naive Bayes,5-fold cross validation + Tf-idf")
plt.plot([100, 500, 1000, 5000, 8000], [0.721872816, 0.762264151, 0.782250175,0.827113906, 0.833682739], label="Naive Bayes Accuracy")
plt.plot([100, 500, 1000, 5000, 8000], [0.734730957,0.780852551, 0.801537386,0.827952481,0.833403215], label="SVM Accuracy")
plt.plot([100, 500, 1000, 5000, 8000], [0.527377648,0.487581633, 0.466636717, 0.415795735, 0.407820133], label="Naive Bayes Mean Square Error")
plt.plot([100, 500, 1000, 5000, 8000], [0.515042758,0.468131872, 0.445491429,0.414786113,0.408162695], label="SVM Mean Square Error")

plt.xlabel("Number of Features")
plt.ylabel("Accuracy an Mean Square error")
plt.legend()
plt.savefig('Comapring SVM and Naive Bayes,5-fold cross validation + Tf-idf')
plt.show()

#Comparing the AUC score:
plt.figure()
plt.title("5-fold cross validation + Tf-idf_Models AUC Comparison")
plt.plot([100, 500, 1000, 5000, 8000], [0.818975,0.864813905,0.885463719,0.911751751,0.915821543], label="Logisic Regression Model")
plt.plot([100, 500, 1000, 5000, 8000], [0.81304184,0.853980666,0.874391643,0.908640443,0.916013129], label="Naive Bayes")
plt.plot([100, 500, 1000, 5000, 8000], [0.81882071181932, 0.8545819119728403, 0.8649514246933474, 0.8757580611062641,0.877073272], label="Random Forest")
plt.plot([100, 500, 1000, 5000, 8000], [0.739610819, 0.78206851,0.801912445,0.827420573,0.833059535], label="SVM")
plt.xlabel("Number of Features")
plt.ylabel("AUC Score")
plt.legend()
plt.savefig('5-fold cross validation + Tf-idf_Models AUC Comparison')
plt.show()