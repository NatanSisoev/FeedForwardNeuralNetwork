OUTPUT_DIR=/home/alumnos/capmc/capmc-1/Escritorio/FFNN-SourceCode/OUT
OUTFILE_PREFIX=slurm-
ERRFILE_PREFIX=slurm-

PYTHON=python3
RUN_SCRIPT=run.py

PARTITIONS := nodo.q new-nodo.q
NPROCS := 2 4 6 8 10 12 16 20 24 32 36 40 48 64
NUM_EPOCHS := 10 100
NUM_NEURONS := 135 250
REPEAT=5

all:
	@jid=$$(sbatch mpi.sub | awk '{print $$4}'); \
	echo "Submitted job $$jid"; \
	while squeue -h -j $$jid | grep -q "$$jid"; do sleep 0.2; done; \
	outf="$(OUTPUT_DIR)/$(OUTFILE_PREFIX)$$jid.out"; \
	errf="$(OUTPUT_DIR)/$(ERRFILE_PREFIX)$$jid.err"; \
	code $$outf

test1:
	@echo "Starting batch runs..."
	@for p in $(PARTITIONS); do \
	  for n in $(NPROCS); do \
		for h in $(NUM_EPOCHS); do \
			for nn in $(NUM_NEURONS); do \
				echo "Running: partition=$$p nprocs=$$n num_epochs=$$h num_neurons=$$nn repeat=$(REPEAT)"; \
				$(PYTHON) $(RUN_SCRIPT) --name test1 --partition $$p --nprocs $$n --num_epochs $$h --num_neurons $$nn --repeat $(REPEAT) --no-open; \
			done; \
		done; \
	  done; \
	done
