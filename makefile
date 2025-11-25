EXEC=exec
PROFILE=profile.nsys-rep.qdrep

OUTPUT_DIR=/home/alumnos/capmc/capmc-1/Escritorio/FFNN-SourceCode/OUT
OUTFILE_PREFIX=out-
ERRFILE_PREFIX=err-

all:
	@jid=$$(sbatch scheduler.sub X | awk '{print $$4}'); \
	echo "Submitted job $$jid"; \
	while squeue -h -j $$jid | grep -q "$$jid"; do sleep 0.2; done; \
	f="$(OUTPUT_DIR)/$(OUTFILE_PREFIX)$$jid.out"; \
	e="$(OUTPUT_DIR)/$(ERRFILE_PREFIX)$$jid.out"; \
	echo "Output file path: $$f"; \
	echo "Errors file path: $$e"; \

clean:
	rm -f $(EXEC) *.o *.out *.qdrep *.sqlite out

squeue:
	tmux new-session \; \
		split-window -h \; \
		select-pane -t 0 \; \
		send-keys "watch -n 0.1 squeue" C-m \; \
		select-pane -t 1

stats:
	nsys stats ./$(PROFILE)
