#!/bin/bash

# Script to run FSDP2 vs DDP correctness tests in parallel
# Tests are run in parallel using GPU pairs (each test uses 2 GPUs)
# Usage: ./run_fsdp_tests.sh [/path/to/results]
#        ./run_fsdp_tests.sh --report /path/to/results
#        ./run_fsdp_tests.sh --test <test_name> [/path/to/results]
#        ./run_fsdp_tests.sh --debug --test <test_name>
#        ./run_fsdp_tests.sh --rerun-failed /path/to/results

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREY='\033[0;90m'
DIM='\033[0;90m'
NC='\033[0m'

GPUS_PER_TEST=2
TEST_FILE="tests/fsdp/test_fsdp_vs_ddp.py"

# Test names match exact pytest node IDs
TEST_NAMES=(
    "test_fsdp2_sharding_structure[untied-2gpus]"
    "test_fsdp2_sharding_structure[tied-2gpus]"
    "test_fsdp2_auto_plan_vs_ddp[float32-untied-2gpus]"
    "test_fsdp2_auto_plan_vs_ddp[bfloat16-untied-2gpus]"
    "test_fsdp2_auto_plan_vs_ddp[float32-tied-2gpus]"
    "test_fsdp2_auto_plan_vs_ddp[bfloat16-tied-2gpus]"
    "test_fsdp2_manual_plan_vs_ddp[float32-untied-2gpus]"
    "test_fsdp2_manual_plan_vs_ddp[bfloat16-untied-2gpus]"
    "test_fsdp2_manual_plan_vs_ddp[float32-tied-2gpus]"
    "test_fsdp2_manual_plan_vs_ddp[bfloat16-tied-2gpus]"
    "test_fsdp2_save_load[2gpus]"
)

# ── Helpers ─────────────────────────────────────────────────────────────────
safe_filename() {
    echo "$1" | tr '[]' '_'
}

# ── Report ──────────────────────────────────────────────────────────────────
print_report() {
    local results_dir=$1
    results_dir=$(cd "$results_dir" && pwd)

    if [ ! -d "$results_dir" ]; then
        echo "Error: Results directory '$results_dir' does not exist"
        exit 1
    fi

    echo "=========================================="
    echo "  FSDP2 vs DDP Test Report"
    echo "  Results directory: $results_dir"
    echo "=========================================="
    echo ""

    local success_count=0
    local fail_count=0
    local skip_count=0
    local missing_count=0

    for test_name in "${TEST_NAMES[@]}"; do
        local safe_name
        safe_name=$(safe_filename "$test_name")
        local result_file="$results_dir/${safe_name}.result"
        if [ -f "$result_file" ]; then
            local result=$(cat "$result_file")
            if [[ "$result" == "SUCCESS" ]]; then
                echo -e "${GREEN}✓ ${test_name}: ${result}${NC}"
                ((success_count++))
            elif [[ "$result" == "SKIPPED" ]]; then
                echo -e "${GREY}○ ${test_name}: ${result}${NC}"
                ((skip_count++))
            else
                echo -e "${RED}✗ ${test_name}: ${result}${NC}"
                if [ -f "$results_dir/${safe_name}.log" ]; then
                    echo -e "${DIM}  Error snippet:"
                    tail -n 5 "$results_dir/${safe_name}.log" | while read -r line; do echo -e "    ${DIM}${line}${NC}"; done
                fi
                ((fail_count++))
            fi
        else
            echo -e "${YELLOW}? ${test_name}: NOT RUN${NC}"
            ((missing_count++))
        fi
    done

    echo ""
    echo "-------------------------------------------"
    echo -e "Total: ${GREEN}${success_count} passed${NC}, ${GREY}${skip_count} skipped${NC}, ${RED}${fail_count} failed${NC}, ${YELLOW}${missing_count} not run${NC}"
    echo "=========================================="

    if [ $fail_count -gt 0 ]; then
        echo ""
        echo "Failed test logs (full paths):"
        for test_name in "${TEST_NAMES[@]}"; do
            local safe_name
            safe_name=$(safe_filename "$test_name")
            result_file="$results_dir/${safe_name}.result"
            if [ -f "$result_file" ] && [ "$(cat "$result_file")" != "SUCCESS" ] && [ "$(cat "$result_file")" != "SKIPPED" ]; then
                echo "  $results_dir/${safe_name}.log"
            fi
        done
        exit 1
    fi
}

# ── Argument parsing ────────────────────────────────────────────────────────
if [ "$1" == "--report" ]; then
    if [ -z "$2" ]; then
        echo "Usage: $0 --report /path/to/results"
        exit 1
    fi
    print_report "$2"
    exit 0
fi

DEBUG_MODE=""
if [ "$1" == "--debug" ]; then
    DEBUG_MODE=1
    shift
fi

SINGLE_TEST=""
if [ "$1" == "--test" ]; then
    if [ -z "$2" ]; then
        echo "Usage: $0 --test <test_name> [/path/to/results]"
        echo "Available tests:"
        printf '  %s\n' "${TEST_NAMES[@]}"
        exit 1
    fi
    SINGLE_TEST="$2"
    found=false
    for name in "${TEST_NAMES[@]}"; do
        [ "$name" == "$SINGLE_TEST" ] && found=true && break
    done
    if [ "$found" == false ]; then
        echo "Error: Unknown test '$SINGLE_TEST'"
        echo "Available tests:"
        printf '  %s\n' "${TEST_NAMES[@]}"
        exit 1
    fi
    shift 2
fi

RERUN_FAILED=""
if [ "$1" == "--rerun-failed" ]; then
    if [ -z "$2" ]; then
        echo "Usage: $0 --rerun-failed /path/to/results"
        exit 1
    fi
    RERUN_FAILED=1
    RESULTS_DIR="$2"
    shift 2
    if [ ! -d "$RESULTS_DIR" ]; then
        echo "Error: Results directory '$RESULTS_DIR' does not exist"
        exit 1
    fi
    RESULTS_DIR=$(cd "$RESULTS_DIR" && pwd)
    FAILED_NAMES=()
    for test_name in "${TEST_NAMES[@]}"; do
        safe_name=$(safe_filename "$test_name")
        result_file="$RESULTS_DIR/${safe_name}.result"
        if [ -f "$result_file" ]; then
            result=$(cat "$result_file")
            if [[ "$result" != "SUCCESS" ]] && [[ "$result" != "SKIPPED" ]]; then
                FAILED_NAMES+=("$test_name")
            fi
        fi
    done
    if [ ${#FAILED_NAMES[@]} -eq 0 ]; then
        echo "No failed tests to rerun in $RESULTS_DIR"
        exit 0
    fi
    TEST_NAMES=("${FAILED_NAMES[@]}")
    echo "Rerunning ${#TEST_NAMES[@]} failed test(s): ${TEST_NAMES[*]}"
fi

# ── GPU detection ───────────────────────────────────────────────────────────
AVAILABLE_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$AVAILABLE_GPUS" -lt "$GPUS_PER_TEST" ]; then
    echo "Need at least $GPUS_PER_TEST GPUs, but only $AVAILABLE_GPUS detected!"
    exit 1
fi
NUM_PARALLEL=$((AVAILABLE_GPUS / GPUS_PER_TEST))
echo "Using $AVAILABLE_GPUS GPUs ($NUM_PARALLEL parallel test slots, $GPUS_PER_TEST GPUs each)"

if [ -n "$DEBUG_MODE" ] && [ -z "$SINGLE_TEST" ]; then
    echo "Error: --debug requires --test <test_name>"
    echo "Usage: $0 --debug --test <test_name>"
    exit 1
fi

if [ -n "$SINGLE_TEST" ]; then
    TEST_NAMES=("$SINGLE_TEST")
    echo "Running single test: $SINGLE_TEST"
fi

# ── Results directory ───────────────────────────────────────────────────────
if [ -n "$RERUN_FAILED" ]; then
    mkdir -p "$RESULTS_DIR"
    CLEANUP_RESULTS=false
elif [ -n "$1" ]; then
    RESULTS_DIR="$1"
    mkdir -p "$RESULTS_DIR"
    CLEANUP_RESULTS=false
elif [ -n "$RESULTS_DIR" ]; then
    mkdir -p "$RESULTS_DIR"
    CLEANUP_RESULTS=false
else
    RESULTS_DIR=$(mktemp -d)
    CLEANUP_RESULTS=true
fi
RESULTS_DIR=$(cd "$RESULTS_DIR" && pwd)

if [ "$CLEANUP_RESULTS" = true ]; then
    trap "rm -rf $RESULTS_DIR" EXIT
fi

echo "Results directory: $RESULTS_DIR"

echo "=========================================="
echo "  FSDP2 vs DDP Tests"
echo "  (Parallel execution: $NUM_PARALLEL tests at a time)"
echo "=========================================="
echo ""

# ── Run a single test on a GPU pair ────────────────────────────────────────
run_test() {
    local test_name=$1
    local slot_id=$2
    local safe_name
    safe_name=$(safe_filename "$test_name")
    local result_file="$RESULTS_DIR/${safe_name}.result"

    local gpu_start=$((slot_id * GPUS_PER_TEST))
    local gpu_end=$((gpu_start + GPUS_PER_TEST - 1))
    local gpu_list=""
    for ((g=gpu_start; g<=gpu_end; g++)); do
        [ -n "$gpu_list" ] && gpu_list+=","
        gpu_list+="$g"
    done

    echo -e "${YELLOW}[GPUs ${gpu_list}] Starting: ${test_name}${NC}"

    CUDA_VISIBLE_DEVICES=$gpu_list \
        python -m pytest -v -rs "${TEST_FILE}::${test_name}" \
        > "$RESULTS_DIR/${safe_name}.log" 2>&1

    local exit_code=$?
    local log_file="$RESULTS_DIR/${safe_name}.log"

    local skipped_only=false
    if [ $exit_code -eq 5 ]; then
        skipped_only=true
    elif [ $exit_code -eq 0 ]; then
        if grep -q "passed" "$log_file"; then
            skipped_only=false
        elif grep -q "skipped" "$log_file"; then
            skipped_only=true
        elif grep -q "deselected" "$log_file" && ! grep -q "passed" "$log_file"; then
            skipped_only=true
        fi
    fi

    if [ "$skipped_only" = true ]; then
        echo "SKIPPED" > "$result_file"
        echo -e "${GREY}○ [GPUs ${gpu_list}] ${test_name}: SKIPPED${NC}"
    elif [ $exit_code -eq 0 ]; then
        echo "SUCCESS" > "$result_file"
        echo -e "${GREEN}✓ [GPUs ${gpu_list}] ${test_name}: SUCCESS${NC}"
    else
        echo "FAILED (exit code: $exit_code)" > "$result_file"
        echo -e "${RED}✗ [GPUs ${gpu_list}] ${test_name}: FAILED (exit code: $exit_code)${NC}"
    fi
}

# ── Debug mode (single test with debugpy) ───────────────────────────────────
if [ -n "$DEBUG_MODE" ]; then
    DEBUGPY_PORT=${DEBUGPY_PORT:-5678}
    echo -e "${YELLOW}Debug mode: launching with debugpy on port ${DEBUGPY_PORT}${NC}"
    echo -e "${YELLOW}Attach your debugger (VS Code / Cursor) to localhost:${DEBUGPY_PORT}, then the test will proceed.${NC}"
    echo ""
    CUDA_VISIBLE_DEVICES=0,1 \
        python -m debugpy --listen 0.0.0.0:${DEBUGPY_PORT} --wait-for-client \
        -m pytest -v -rs -s "${TEST_FILE}::${TEST_NAMES[0]}"
    exit $?
fi

# ── Parallel dispatch ───────────────────────────────────────────────────────
declare -a PIDS=()
declare -A PID_SLOT=()

next_free_slot() {
    local used_slots=()
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            used_slots+=("${PID_SLOT[$pid]}")
        fi
    done
    for ((s=0; s<NUM_PARALLEL; s++)); do
        local in_use=false
        for u in "${used_slots[@]}"; do
            if [ "$u" -eq "$s" ]; then
                in_use=true
                break
            fi
        done
        if [ "$in_use" = false ]; then
            echo "$s"
            return
        fi
    done
    echo "-1"
}

for test_name in "${TEST_NAMES[@]}"; do
    # Wait until a slot is free
    while true; do
        slot=$(next_free_slot)
        if [ "$slot" -ge 0 ]; then
            break
        fi
        wait -n 2>/dev/null || sleep 0.5
        # Prune finished PIDs
        NEW_PIDS=()
        for pid in "${PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                NEW_PIDS+=("$pid")
            else
                unset PID_SLOT[$pid]
            fi
        done
        PIDS=("${NEW_PIDS[@]}")
    done

    run_test "$test_name" "$slot" &
    pid=$!
    PIDS+=($pid)
    PID_SLOT[$pid]=$slot
done

echo ""
echo "Waiting for all tests to complete..."
wait

# ── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "  SUMMARY"
echo "=========================================="
echo ""

success_count=0
fail_count=0
skip_count=0

for test_name in "${TEST_NAMES[@]}"; do
    safe_name=$(safe_filename "$test_name")
    result_file="$RESULTS_DIR/${safe_name}.result"
    if [ -f "$result_file" ]; then
        result=$(cat "$result_file")
        if [[ "$result" == "SUCCESS" ]]; then
            echo -e "${GREEN}✓ ${test_name}: ${result}${NC}"
            ((success_count++))
        elif [[ "$result" == "SKIPPED" ]]; then
            echo -e "${GREY}○ ${test_name}: ${result}${NC}"
            ((skip_count++))
        else
            echo -e "${RED}✗ ${test_name}: ${result}${NC}"
            echo -e "${DIM}  Error snippet:"
            tail -n 5 "$RESULTS_DIR/${safe_name}.log" | while read -r line; do echo -e "    ${DIM}${line}${NC}"; done
            ((fail_count++))
        fi
    else
        echo -e "${RED}✗ ${test_name}: NO RESULT (test may have crashed)${NC}"
        ((fail_count++))
    fi
done

echo ""
echo "-------------------------------------------"
echo -e "Total: ${GREEN}${success_count} passed${NC}, ${GREY}${skip_count} skipped${NC}, ${RED}${fail_count} failed${NC}"
echo "=========================================="

if [ $fail_count -gt 0 ]; then
    echo ""
    echo "Failed test logs (full paths):"
    for test_name in "${TEST_NAMES[@]}"; do
        safe_name=$(safe_filename "$test_name")
        result_file="$RESULTS_DIR/${safe_name}.result"
        if [ -f "$result_file" ] && [ "$(cat "$result_file")" != "SUCCESS" ] && [ "$(cat "$result_file")" != "SKIPPED" ]; then
            echo "  $RESULTS_DIR/${safe_name}.log"
        fi
    done
fi

if [ $fail_count -gt 0 ]; then
    exit 1
fi
