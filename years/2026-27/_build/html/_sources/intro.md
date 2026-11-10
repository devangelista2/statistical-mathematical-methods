# Statistical and Mathematical Methods for Machine Learning (SMM)
What follows are the teaching materials for the Statistical and Mathematical Methods for Machine Learning (SMM, in the following), Module 2, for the Academic Year 2026-2027, held by professor Davide Evangelista.

## Topics
Module 2 of the SMM course is dedicated to the **optimization algorithms** used to train Machine Learning models, namely **Gradient Descent**, **Stochastic Gradient Descent (SGD)** and **Adam**. To make the discussion concrete, all the algorithms will be applied to the training of (simple) **Neural Networks**, which we will build from scratch using `numpy`. The course is divided into five parts:

1. **A (Very Short) Introduction to Machine Learning**: what are Artificial Intelligence, Machine Learning, Deep Learning and Data Science; supervised and unsupervised learning; regression and classification; the Machine Learning pipeline.
2. **Neural Networks for Classification**: how to build a classifier as a graph of simple computational units (the neurons), up to the Multi-Layer Perceptron (MLP); binary classification with the sigmoid and the Binary Cross-Entropy loss; how to compute the gradient of the loss via **backpropagation**.
3. **Gradient Descent**: optimization problems, convexity, optimality conditions, the Gradient Descent algorithm and its convergence, the choice of the step size, backtracking and stopping criteria.
4. **Stochastic Gradient Descent**: the finite-sum structure of the loss, mini-batches, iterations and epochs, the variance of the stochastic gradient and the choice of the learning rate.
5. **Adam and a Realistic Experiment**: momentum, adaptive learning rates and Adam; the full training of a Neural Network on a real image classification dataset.

The focus will always be to deeply understand the math behind those concepts, and how their implementation relates to the basic linear algebra and calculus concepts you studied during your bachelor degree.

## Exam

### Students enrolled in A.Y. 2026-2027
The exam of this module consists of a short **homework**: a small project, to be solved in a Jupyter notebook, in which you have to apply what we studied during the course to a real classification problem, commenting and discussing the results. The text of the homework is available in the [Homework](Exam/homework) section. The exam works as follows:

- The homework is **the same for all the students**.
- You can send your solution by e-mail **at any time**. The received homeworks are corrected and graded in correspondence with the **exam dates published on AlmaEsami**.
- The homework is graded based on the correctness of the proposed solution, on the **effort** it shows, and on the ability of the student to **test** and **experiment** with the methods studied in the course. The grade of this module is then combined with the grade of Module 1 to obtain the final grade of the course.

### Students enrolled in A.Y. 2025-2026
For the students enrolled in the A.Y. 2025-2026, a transitional exam modality, similar to the one of the last year, is available (see the [dedicated page](Exam/exam_2025_26) for the details):

- The exam consists of a **quiz only**, taken **in the classroom** on the exam dates published on **AlmaEsami**, containing questions related both to the topics of the quiz and to the topics of the oral exam of the last year.
- **No homework** is required.
- The quiz is the **complete exam** of the course, since it covers the whole program of the A.Y. 2025-2026: it is **not** necessary to take also the exam of Module 1.
- This modality is available **only during the first semester**, i.e. in the **two exam sessions of the winter session**. After that, the exam must be taken with the same modality as all the other students.

## Office hour
The office hour is Wednesday morning, from 9a.m. to 12p.m., via Microsoft Teams or in my office, at Ex Veneta, close to the Zanolini train station. 

Either cases, it is **mandatory** to set an appointment by e-mail, via the address ``davide.evangelista5@unibo.it``.
