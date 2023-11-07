import numpy as np


def PCG(A, b,  Pinv, guess, exit_tolerance = 1e-6, max_iter = 100, DEBUG_MODE = False):
	# initialize
	x = np.reshape(guess, (guess.shape[0],1))
	r = b - np.matmul(A, x)
	
	r_tilde = np.matmul(Pinv, r)
	p = r_tilde
	nu = np.matmul(r.transpose(), r_tilde)
	if DEBUG_MODE:
		print("Initial nu[", nu, "]")
	# loop
	for iteration in range(max_iter):
		if((r == 0).all()):
			break
		Ap = np.matmul(A, p)
		alpha = nu / np.matmul(p.transpose(), Ap)
		r -= alpha * Ap
		x += alpha * p
		
		r_tilde = np.matmul(Pinv, r)
		nu_prime = np.matmul(r.transpose(), r_tilde)
		if abs(nu_prime) < exit_tolerance:
			if DEBUG_MODE:
				print(Pinv)
				print("Exiting with err[", abs(nu_prime), "]")
			break
		else:
			if DEBUG_MODE:
				print("Iter[", iteration, "] with err[", abs(nu_prime), "]")
		
		beta = nu_prime / nu
		p = r_tilde + beta * p
		nu = nu_prime

	return x

def computePreconditioner( A, preconditioner_type, nx=1):
	if preconditioner_type == "0": # null aka identity
		return np.identity(A.shape[0])

	if preconditioner_type == "J": # Jacobi aka Diagonal
		return np.linalg.inv(np.diag(np.diag(A)))

	elif preconditioner_type == "BJ": # Block-Jacobi

		n_blocks = int(A.shape[0] / nx)
		Pinv = np.zeros(A.shape)

		for k in range(n_blocks):
			rc_k = k*nx
			rc_kp1 = rc_k + nx
			Pinv[rc_k:rc_kp1, rc_k:rc_kp1] = np.linalg.inv(A[rc_k:rc_kp1, rc_k:rc_kp1])

		return Pinv

	elif preconditioner_type == "S" or preconditioner_type == "SS": # Stair (for blocktridiagonal of blocksize nq+nv)

		n_blocks = int(A.shape[0] / nx)
		Pinv = np.zeros(A.shape)
		# compute stair inverse
		for k in range(n_blocks):
			# compute the diagonal term
			Pinv[k*nx:(k+1)*nx, k*nx:(k+1)*nx] = np.linalg.inv(A[k*nx:(k+1)*nx, k*nx:(k+1)*nx])
			if np.mod(k, 2): # odd block includes off diag terms
				# Pinv_left_of_diag_k = -Pinv_diag_k * A_left_of_diag_k * -Pinv_diag_km1
				Pinv[k*nx:(k+1)*nx, (k-1)*nx:k*nx] = -np.matmul(Pinv[k*nx:(k+1)*nx, k*nx:(k+1)*nx], \
														np.matmul(A[k*nx:(k+1)*nx, (k-1)*nx:k*nx], \
																Pinv[(k-1)*nx:k*nx, (k-1)*nx:k*nx]))
			elif k > 0: # compute the off diag term for previous odd block (if it exists)
				# Pinv_right_of_diag_km1 = -Pinv_diag_km1 * A_right_of_diag_km1 * -Pinv_diag_k
				Pinv[(k-1)*nx:k*nx, k*nx:(k+1)*nx] = -np.matmul(Pinv[(k-1)*nx:k*nx, (k-1)*nx:k*nx], \
														np.matmul(A[(k-1)*nx:k*nx, k*nx:(k+1)*nx], \
																Pinv[k*nx:(k+1)*nx, k*nx:(k+1)*nx]))
		# make symmetric
		if preconditioner_type == "SS":
			for k in range(n_blocks):
				if np.mod(k, 2): # copy from odd blocks
					# always copy up the left to previous right
					Pinv[(k-1)*nx:k*nx, k*nx:(k+1)*nx] = Pinv[k*nx:(k+1)*nx, (k-1)*nx:k*nx].transpose()
					# if not last block copy right to next left
					if k < n_blocks - 1:
						Pinv[(k+1)*nx:(k+2)*nx, k*nx:(k+1)*nx] = Pinv[k*nx:(k+1)*nx, (k+1)*nx:(k+2)*nx].transpose()
		return Pinv
	else:
		print("Invalid preconditioner options are [J : Jacobi, BJ: Block-Jacobi, AB : Alternating Block, OB : Overlapping Block, S : Stair, SS: Symmetric Stair]")
		exit()

def solve(A, b, preconditioner_type="0", nx=1):
    Pinv = computePreconditioner(A,preconditioner_type, nx)
    guess = np.zeros(A.shape[0])
    x = PCG(A, b, Pinv, guess)
    return x