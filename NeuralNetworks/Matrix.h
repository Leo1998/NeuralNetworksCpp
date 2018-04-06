#pragma once

#include <iostream>

class Matrix {
public:
	Matrix(int, int);
	Matrix();
	~Matrix();
	Matrix(int, int, double);
	Matrix(const Matrix&);

	Matrix& operator=(const Matrix&);

	inline double& operator()(int x, int y) const { return p[x][y]; }

	inline int getRows() const { return rows; }
	inline int getCols() const { return cols; }

	Matrix& operator+=(const Matrix&);
	Matrix& operator-=(const Matrix&);
	Matrix& operator*=(const Matrix&);
	Matrix& operator*=(double);
	Matrix& operator/=(double);

	friend std::ostream& operator<<(std::ostream&, const Matrix&);
	friend std::istream& operator>>(std::istream&, Matrix&);

	void swapRows(int, int);
	void fill(double);
	void randomize(double, double);
	void multElementWise(const Matrix&);
	Matrix transpose() const;
	double sum() const;

	// functions on vectors
	static double dotProduct(Matrix, Matrix);

private:
	int rows, cols;
	double **p;

	void allocSpace();
};

Matrix operator+(const Matrix&, const Matrix&);
Matrix operator-(const Matrix&, const Matrix&);
Matrix operator*(const Matrix&, const Matrix&);
Matrix operator*(const Matrix&, double);
Matrix operator*(double, const Matrix&);
Matrix operator/(const Matrix&, double);