# NMAKE makefile for lychrel. x64 release only.
#
#   nmake            build
#   nmake diag       build the diagnostic binary (span/serial/variance timers)
#   nmake run        build and run against 196
#   nmake clean      remove build output
#   nmake distclean  also remove .isf checkpoints in the repo root and bin\
#
# Run from an "x64 Native Tools Command Prompt for VS", or invoke vcvars64.bat first, so cl.exe and the x64 libraries
# are on the path.
#
# This is enforced, not merely requested - see checkarch below. An x86 prompt builds this source without any error
# (/arch:AVX512 is accepted for 32-bit targets), so the only symptom is a binary that is roughly 3.6x slower because the
# packed core's 64-bit limb arithmetic gets split across register pairs. Measured interleaved in one session at 524288
# digits with 1 thread: x64 3.41e-11 and 3.33e-11 against x86 1.19e-10 and 1.24e-10. That failure is silent and
# correctness check, so it is worth two guards: the environment check before compiling, and a check of the actual PE
# machine type after linking in case the environment is misreporting.

BINDIR = bin
OBJDIR = obj

# /GL plus /LTCG at link. The usual reason for whole-program optimization - inlining across translation units - does
# not apply here, because the project is a single .cpp and the compiler already sees everything. It is enabled for the
# smaller 1-TU effect instead: link-time codegen can give internal static functions non-standard calling conventions
# and drop more dead code. Worth 2.3% on the reference machine (Azure Standard_D16ds_v5, 8 physical Ice Lake-SP cores)
# at 731215 digits with 8 threads - small enough to sit inside the run-to-run spread of a single pair, so re-measure it
# over several paired runs rather than one.

CFLAGS = /nologo /std:c++20 /EHsc /W4 /permissive- /arch:AVX512 /O2 /GL /Gy /DNDEBUG
LDFLAGS = /nologo /LTCG

TARGET = $(BINDIR)\lychrel.exe
OBJ    = $(OBJDIR)\lychrel.obj

DIAGTARGET = $(BINDIR)\lychrel-diag.exe
DIAGOBJ    = $(OBJDIR)\lychrel-diag.obj

all: $(TARGET)

diag: $(DIAGTARGET)

# Refuse to build from anything but an x64 toolchain. VSCMD_ARG_TGT_ARCH is set by vcvars/the Native Tools prompt; an
# undefined value expands to empty here and fails the comparison, which is the intended behaviour for a bare shell that
# has no VS environment at all.
checkarch:
	@if not "$(VSCMD_ARG_TGT_ARCH)" == "x64" echo *** ERROR: toolchain target is "$(VSCMD_ARG_TGT_ARCH)", expected x64.
	@if not "$(VSCMD_ARG_TGT_ARCH)" == "x64" echo *** Run vcvars64.bat, or use the x64 Native Tools Command Prompt.
	@if not "$(VSCMD_ARG_TGT_ARCH)" == "x64" exit 1

$(TARGET): checkarch $(OBJ)
	@if not exist "$(BINDIR)" mkdir "$(BINDIR)"
	link $(LDFLAGS) /OUT:$@ $(OBJ)
	@dumpbin /nologo /headers $@ | findstr /c:"machine (x64)" >nul || (echo *** ERROR: $@ is not an x64 image. & exit 1)

$(OBJ): checkarch lychrel.cpp
	@if not exist "$(OBJDIR)" mkdir "$(OBJDIR)"
	cl $(CFLAGS) /c lychrel.cpp /Fo:$@ /Fd:$(OBJDIR)\lychrel.pdb

$(DIAGTARGET): checkarch $(DIAGOBJ)
	@if not exist "$(BINDIR)" mkdir "$(BINDIR)"
	link $(LDFLAGS) /OUT:$@ $(DIAGOBJ)
	@dumpbin /nologo /headers $@ | findstr /c:"machine (x64)" >nul || (echo *** ERROR: $@ is not an x64 image. & exit 1)

$(DIAGOBJ): checkarch lychrel.cpp
	@if not exist "$(OBJDIR)" mkdir "$(OBJDIR)"
	cl $(CFLAGS) /DLYCHREL_DIAG /c lychrel.cpp /Fo:$@ /Fd:$(OBJDIR)\lychrel-diag.pdb

run: $(TARGET)
	$(TARGET) 196

# Removes build output only. Deliberately deletes the build artifacts inside $(BINDIR) rather than the directory
# itself, because runs launched from bin\ leave checkpoints there and rmdir would take them too.
clean:
	@if exist $(BINDIR) del /q $(BINDIR)\*.exe $(BINDIR)\*.pdb $(BINDIR)\*.ilk 2>nul
	@if exist $(OBJDIR) rmdir /s /q $(OBJDIR)
	@del /q *.obj *.exe *.pdb *.ilk 2>nul

# Checkpoints are written to whatever directory the run was launched from, so they collect in the repo root and in
# bin\ and are easily gigabytes. Kept out of clean because a checkpoint can represent hours of search: deleting one has
# to be asked for, not implied by a build tidy-up.
distclean: clean
	@del /q *.isf 2>nul
	@if exist $(BINDIR) del /q $(BINDIR)\*.isf 2>nul
	@if exist $(BINDIR) rmdir $(BINDIR) 2>nul
