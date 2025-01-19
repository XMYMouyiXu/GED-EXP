package ged;

public class MathUtil {
	public static void printMatrix(int[][] m) {
	    for (int i = 0; i < m.length; i++) {
	        for (int j = 0; j < m[0].length; j++) {
	            System.out.printf("%4d", m[i][j]);
	        }
	        System.out.println();
	    }
	    System.out.println();
	}
	public static void printMatrix(double[][] m) {
	    for (int i = 0; i < m.length; i++) {
	        for (int j = 0; j < m[0].length; j++) {
	            System.out.print(m[i][j] + " ");
	        }
	        System.out.println();
	    }
	    System.out.println();
	}
	public static int[][] matrixPower(int[][] m, int k) {
		if(m.length == 0)
		{
			return m;
		}
	    int[][] result = new int[m.length][m[0].length];
	    //initialize result as identity matrix
	    for (int i = 0; i < m.length; i++) {
	        result[i][i] = 1;
	    }
	    //compute m^k using binary exponentiation
	    while (k > 0) {
	        if (k % 2 == 1) {
	            result = matrixMultiply(result, m);
	        }
	        m = matrixMultiply(m, m);
	        k /= 2;
	    }
	    return result;
	}

	public static int[][] matrixMultiply(int[][] a, int[][] b) {
	    int[][] c = new int[a.length][b[0].length];
	    for (int i = 0; i < a.length; i++) {
	        for (int j = 0; j < b[0].length; j++) {
	            for (int k = 0; k < b.length; k++) {
	                c[i][j] += a[i][k] * b[k][j];
	            }
	        }
	    }
	    return c;
	}
}
