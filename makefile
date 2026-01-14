OUTPUT_DIR=/home/alumnos/capmc/capmc-1/Escritorio/FFNN-SourceCode/OUT
OUTFILE_PREFIX=slurm-
ERRFILE_PREFIX=slurm-

all:
	@jid=$$(sbatch mpi.sub | awk '{print $$4}'); \
	echo "Submitted job $$jid"; \
	while squeue -h -j $$jid | grep -q "$$jid"; do sleep 0.2; done; \
	outf="$(OUTPUT_DIR)/$(OUTFILE_PREFIX)$$jid.out"; \
	errf="$(OUTPUT_DIR)/$(ERRFILE_PREFIX)$$jid.err"; \
	echo "Output file: $$outf"; \
	echo "Error file: $$errf"; \
	code $$outf
