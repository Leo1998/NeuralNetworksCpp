#include "Matrix.h"

#include <iostream>
#include "MathUtil.h"

namespace nn {

	Matrix::Matrix(int rows, int cols) : rows(rows), cols(cols)
	{
		allocSpace();
		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < cols; ++j) {
				p[i][j] = 0;
			}
		}
	}

	Matrix::Matrix() : rows(1), cols(1)
	{
		allocSpace();
		p[0][0] = 0;
	}

	Matrix::~Matrix()
	{
		for (int i = 0; i < rows; ++i) {
			delete[] p[i];
		}
		delete[] p;
	}

	Matrix::Matrix(int rows, int cols, double diagonal) : rows(rows), cols(cols)
	{
		allocSpace();
		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < cols; ++j) {
				if (i == j) {
					p[i][j] = diagonal;
				}
				else {
					p[i][j] = 0;
				}
			}
		}
	}

	Matrix::Matrix(const Matrix& m) : rows(m.rows), cols(m.cols)
	{
		allocSpace();
		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < cols; ++j) {
				p[i][j] = m.p[i][j];
			}
		}
	}

	Matrix& Matrix::operator=(const Matrix& m)
	{
		if (this == &m) {
			return *this;
		}

		if (rows != m.rows || cols != m.cols) {
			for (int i = 0; i < rows; ++i) {
				delete[] p[i];
			}
			delete[] p;

			rows = m.rows;
			cols = m.cols;
			allocSpace();
		}

		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < cols; ++j) {
				p[i][j] = m.p[i][j];
			}
		}
		return *this;
	}

	Matrix& Matrix::operator+=(const Matrix& m)
	{
		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < cols; ++j) {
				p[i][j] += m.p[i][j];
			}
		}
		return *this;
	}

	Matrix& Matrix::operator-=(const Matrix& m)
	{
		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < cols; ++j) {
				p[i][j] -= m.p[i][j];
			}
		}
		return *this;
	}

	Matrix& Matrix::operator*=(const Matrix& m)
	{
		if (cols != m.rows)
			throw 69;

		Matrix temp(rows, m.cols);
		for (int i = 0; i < temp.rows; ++i) {
			for (int j = 0; j < temp.cols; ++j) {
				for (int k = 0; k < cols; ++k) {
					temp.p[i][j] += (p[i][k] * m.p[k][j]);
				}
			}
		}
		return (*this = temp);
	}

	Matrix& Matrix::operator*=(double num)
	{
		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < cols; ++j) {
				p[i][j] *= num;
			}
		}
		return *this;
	}

	Matrix& Matrix::operator/=(double num)
	{
		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < cols; ++j) {
				p[i][j] /= num;
			}
		}
		return *this;
	}

	void Matrix::swapRows(int r1, int r2)
	{
		double *temp = p[r1];
		p[r1] = p[r2];
		p[r2] = temp;
	}

	void Matrix::fill(double value)
	{
		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < cols; ++j) {
				p[i][j] = value;
			}
		}
	}

	void Matrix::randomize(double min, double max)
	{
		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < cols; ++j) {
				p[i][j] = randomDouble(min, max);
			}
		}
	}

	void Matrix::multElementWise(const Matrix& m)
	{
		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < cols; ++j) {
				p[i][j] *= m.p[i][j];
			}
		}
	}

	Matrix Matrix::transpose() const
	{
		Matrix ret(cols, rows);
		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < cols; ++j) {
				ret.p[j][i] = p[i][j];
			}
		}
		return ret;
	}

	double Matrix::sum() const
	{
		double sum = 0.0;
		for (int i = 0; i < rows; ++i) {
			for (int j = 0; j < cols; ++j) {
				sum += p[i][j];
			}
		}
		return sum;
	}

	// functions on VECTORS
	double Matrix::dotProduct(Matrix a, Matrix b)
	{
		double sum = 0;
		for (int i = 0; i < a.rows; ++i) {
			sum += (a(i, 0) * b(i, 0));
		}
		return sum;
	}

	/* PRIVATE HELPER FUNCTIONS
	********************************/

	void Matrix::allocSpace()
	{
		p = new double*[rows];
		for (int i = 0; i < rows; ++i) {
			p[i] = new double[cols];
		}
	}

	/* NON-MEMBER FUNCTIONS
	********************************/

	Matrix operator+(const Matrix& m1, const Matrix& m2)
	{
		Matrix temp(m1);
		return (temp += m2);
	}

	Matrix operator-(const Matrix& m1, const Matrix& m2)
	{
		Matrix temp(m1);
		return (temp -= m2);
	}

	Matrix operator*(const Matrix& m1, const Matrix& m2)
	{
		Matrix temp(m1);
		return (temp *= m2);
	}

	Matrix operator*(const Matrix& m, double num)
	{
		Matrix temp(m);
		return (temp *= num);
	}

	Matrix operator*(double num, const Matrix& m)
	{
		return (m * num);
	}

	Matrix operator/(const Matrix& m, double num)
	{
		Matrix temp(m);
		return (temp /= num);
	}

	std::ostream& operator<<(std::ostream& os, const Matrix& m)
	{
		for (int i = 0; i < m.rows; ++i) {
			os << m.p[i][0];
			for (int j = 1; j < m.cols; ++j) {
				os << " " << m.p[i][j];
			}
			os << std::endl;
		}
		return os;
	}

	std::istream& operator>>(std::istream& is, Matrix& m)
	{
		for (int i = 0; i < m.rows; ++i) {
			for (int j = 0; j < m.cols; ++j) {
				is >> m.p[i][j];
			}
		}
		return is;
	}
}