#!/bin/bash

# Script to run tensor parallel (TP) tests for MoE models
# Tests are run in parallel using GPU pairs (each TP test uses 2 GPUs)
# Usage: ./run_moe_tests.sh [/path/to/results]
#        ./run_moe_tests.sh --report /path/to/results
#        ./run_moe_tests.sh --model <model_name> [/path/to/results]
#        ./run_moe_tests.sh --rerun-failed /path/to/results

# Define colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREY='\033[0;90m'
DIM='\033[0;90m'
NC='\033[0m' # No Color

# Number of GPUs required per TP test
GPUS_PER_TEST=2

# Define models to test (model_name -> test_file)
declare -A MODELS=(
    ["afmoe"]="tests/models/afmoe/test_modeling_afmoe.py"
    ["aria"]="tests/models/aria/test_modeling_aria.py"
    ["dbrx"]="tests/models/dbrx/test_modeling_dbrx.py"
    ["deepseek_v2"]="tests/models/deepseek_v2/test_modeling_deepseek_v2.py"
    ["deepseek_v3"]="tests/models/deepseek_v3/test_modeling_deepseek_v3.py"
    ["dots1"]="tests/models/dots1/test_modeling_dots1.py"
    ["ernie4_5_moe"]="tests/models/ernie4_5_moe/test_modeling_ernie4_5_moe.py"
    ["ernie4_5_vl_moe"]="tests/models/ernie4_5_vl_moe/test_modeling_ernie4_5_vl_moe.py"
    ["flex_olmo"]="tests/models/flex_olmo/test_modeling_flex_olmo.py"
    ["glm_moe_dsa"]="tests/models/glm_moe_dsa/test_modeling_glm_moe_dsa.py"
    ["glm4_moe"]="tests/models/glm4_moe/test_modeling_glm4_moe.py"
    ["glm4_moe_lite"]="tests/models/glm4_moe_lite/test_modeling_glm4_moe_lite.py"
    ["glm4v_moe"]="tests/models/glm4v_moe/test_modeling_glm4v_moe.py"
    ["gpt_oss"]="tests/models/gpt_oss/test_modeling_gpt_oss.py"
    ["granitemoe"]="tests/models/granitemoe/test_modeling_granitemoe.py"
    ["granitemoehybrid"]="tests/models/granitemoehybrid/test_modeling_granitemoehybrid.py"
    ["granitemoeshared"]="tests/models/granitemoeshared/test_modeling_granitemoeshared.py"
    ["hunyuan_v1_moe"]="tests/models/hunyuan_v1_moe/test_modeling_hunyuan_v1_moe.py"
    ["jamba"]="tests/models/jamba/test_modeling_jamba.py"
    ["jetmoe"]="tests/models/jetmoe/test_modeling_jetmoe.py"
    ["lfm2_moe"]="tests/models/lfm2_moe/test_modeling_lfm2_moe.py"
    ["llama4"]="tests/models/llama4/test_modeling_llama4.py"
    ["longcat_flash"]="tests/models/longcat_flash/test_modeling_longcat_flash.py"
    ["minimax"]="tests/models/minimax/test_modeling_minimax.py"
    ["minimax_m2"]="tests/models/minimax_m2/test_modeling_minimax_m2.py"
    ["mixtral"]="tests/models/mixtral/test_modeling_mixtral.py"
    ["nllb_moe"]="tests/models/nllb_moe/test_modeling_nllb_moe.py"
    ["olmoe"]="tests/models/olmoe/test_modeling_olmoe.py"
    ["phimoe"]="tests/models/phimoe/test_modeling_phimoe.py"
    ["qwen2_moe"]="tests/models/qwen2_moe/test_modeling_qwen2_moe.py"
    ["qwen3_moe"]="tests/models/qwen3_moe/test_modeling_qwen3_moe.py"
    ["qwen3_next"]="tests/models/qwen3_next/test_modeling_qwen3_next.py"
    ["qwen3_omni_moe"]="tests/models/qwen3_omni_moe/test_modeling_qwen3_omni_moe.py"
    ["qwen3_vl_moe"]="tests/models/qwen3_vl_moe/test_modeling_qwen3_vl_moe.py"
    ["qwen3_5_moe"]="tests/models/qwen3_5_moe/test_modeling_qwen3_5_moe.py"
    ["solar_open"]="tests/models/solar_open/test_modeling_solar_open.py"
    ["switch_transformers"]="tests/models/switch_transformers/test_modeling_switch_transformers.py"
)""

# Get model names array
MODEL_NAMES=(${!MODELS[@]})

# Report function - print summary from existing results directory
print_report() {
    local results_dir=$1
    results_dir=$(cd "$results_dir" && pwd)  # absolute path for clickable links

    if [ ! -d "$results_dir" ]; then
        echo "Error: Results directory '$results_dir' does not exist"
        exit 1
    fi

    echo "=========================================="
    echo "  MoE Models TP Test Report"
    echo "  Results directory: $results_dir"
    echo "=========================================="
    echo ""
    
    local success_count=0
    local fail_count=0
    local skip_count=0
    local missing_count=0
    
    for model_name in "${MODEL_NAMES[@]}"; do
        local result_file="$results_dir/${model_name}.result"
        if [ -f "$result_file" ]; then
            local result=$(cat "$result_file")
            if [[ "$result" == "SUCCESS" ]]; then
                echo -e "${GREEN}✓ ${model_name}: ${result}${NC}"
                ((success_count++))
            elif [[ "$result" == "SKIPPED" ]]; then
                echo -e "${GREY}○ ${model_name}: ${result}${NC}"
                ((skip_count++))
            else
                echo -e "${RED}✗ ${model_name}: ${result}${NC}"
                # Show last few lines of error
                if [ -f "$results_dir/${model_name}.log" ]; then
                    echo -e "${DIM}  Error snippet:"
                    tail -n 5 "$results_dir/${model_name}.log" | while read -r line; do echo -e "    ${DIM}${line}${NC}"; done
                fi
                ((fail_count++))
            fi
        else
            echo -e "${YELLOW}? ${model_name}: NOT RUN${NC}"
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
        for model_name in "${MODEL_NAMES[@]}"; do
            result_file="$results_dir/${model_name}.result"
            if [ -f "$result_file" ] && [ "$(cat "$result_file")" != "SUCCESS" ] && [ "$(cat "$result_file")" != "SKIPPED" ]; then
                echo "  $results_dir/${model_name}.log"
            fi
        done
        exit 1
    fi
}

# Handle --report argument
if [ "$1" == "--report" ]; then
    if [ -z "$2" ]; then
        echo "Usage: $0 --report /path/to/results"
        exit 1
    fi
    print_report "$2"
    exit 0
fi

# Handle --model argument (run single model test)
SINGLE_MODEL=""
if [ "$1" == "--model" ]; then
    if [ -z "$2" ]; then
        echo "Usage: $0 --model <model_name> [/path/to/results]"
        echo "Available models: ${MODEL_NAMES[*]}"
        exit 1
    fi
    SINGLE_MODEL="$2"
    # Validate model name exists
    if [ -z "${MODELS[$SINGLE_MODEL]}" ]; then
        echo "Error: Unknown model '$SINGLE_MODEL'"
        echo "Available models: ${MODEL_NAMES[*]}"
        exit 1
    fi
    shift 2  # Remove --model and model_name from arguments
fi

# Handle --rerun-failed argument (rerun only failed tests from a previous run)
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
    for model_name in "${MODEL_NAMES[@]}"; do
        result_file="$RESULTS_DIR/${model_name}.result"
        if [ -f "$result_file" ]; then
            result=$(cat "$result_file")
            if [[ "$result" != "SUCCESS" ]] && [[ "$result" != "SKIPPED" ]]; then
                FAILED_NAMES+=("$model_name")
            fi
        fi
    done
    if [ ${#FAILED_NAMES[@]} -eq 0 ]; then
        echo "No failed tests to rerun in $RESULTS_DIR"
        exit 0
    fi
    MODEL_NAMES=("${FAILED_NAMES[@]}")
    echo "Rerunning ${#MODEL_NAMES[@]} failed test(s): ${MODEL_NAMES[*]}"
fi

# Check available GPUs and calculate parallel slots
AVAILABLE_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$AVAILABLE_GPUS" -lt "$GPUS_PER_TEST" ]; then
    echo "Need at least $GPUS_PER_TEST GPUs for TP tests, but only $AVAILABLE_GPUS detected!"
    exit 1
fi
NUM_PARALLEL=$((AVAILABLE_GPUS / GPUS_PER_TEST))
echo "Using $AVAILABLE_GPUS GPUs ($NUM_PARALLEL parallel test slots, $GPUS_PER_TEST GPUs each)"

# If single model mode, override MODEL_NAMES to only include that model
if [ -n "$SINGLE_MODEL" ]; then
    MODEL_NAMES=("$SINGLE_MODEL")
    echo "Running single model test: $SINGLE_MODEL"
fi

# Handle results directory - use provided path or create temp directory
if [ -n "$RERUN_FAILED" ]; then
    mkdir -p "$RESULTS_DIR"
    CLEANUP_RESULTS=false
elif [ -n "$1" ]; then
    RESULTS_DIR="$1"
    mkdir -p "$RESULTS_DIR"
    CLEANUP_RESULTS=false
elif [ -n "$RESULTS_DIR" ]; then
    # RESULTS_DIR already set via environment variable
    mkdir -p "$RESULTS_DIR"
    CLEANUP_RESULTS=false
else
    RESULTS_DIR=$(mktemp -d)
    CLEANUP_RESULTS=true
fi
# Resolve to absolute path for clickable links in terminal
RESULTS_DIR=$(cd "$RESULTS_DIR" && pwd)

# Only cleanup if we created a temp directory
if [ "$CLEANUP_RESULTS" = true ]; then
    trap "rm -rf $RESULTS_DIR" EXIT
fi

echo "Results directory: $RESULTS_DIR"

echo "=========================================="
echo "  MoE Models TP Test Script"
echo "  (Parallel execution: $NUM_PARALLEL tests at a time)"
echo "=========================================="
echo ""

# Function to run TP pytest tests on a specific GPU pair
run_test() {
    local model_name=$1
    local test_file=$2
    local slot_id=$3
    local result_file="$RESULTS_DIR/${model_name}.result"
    
    # Calculate GPU pair for this slot (slot 0 -> GPUs 0,1; slot 1 -> GPUs 2,3; etc.)
    local gpu_start=$((slot_id * GPUS_PER_TEST))
    local gpu_end=$((gpu_start + GPUS_PER_TEST - 1))
    local gpu_list="${gpu_start},${gpu_end}"
    
    echo -e "${YELLOW}[GPUs ${gpu_list}] Starting: ${model_name}${NC}"
    
    # Run only tensor parallel tests from TensorParallelTesterMixin
    # Specifically: test_tp_forward_direct, test_tp_backward_direct, test_tp_generation_direct, test_tp_generation_with_conversion
    CUDA_VISIBLE_DEVICES=$gpu_list \
        python -m pytest -v -rs "$test_file" -k "test_tp_forward or test_tp_backward or test_tp_generation" \
        > "$RESULTS_DIR/${model_name}.log" 2>&1
    
    local exit_code=$?
    local log_file="$RESULTS_DIR/${model_name}.log"
    
    # Check if all tests were skipped or deselected
    local skipped_only=false
    # Exit code 5 = no tests collected (all deselected)
    if [ $exit_code -eq 5 ]; then
        skipped_only=true
    elif [ $exit_code -eq 0 ]; then
        # Check if there were any passed tests or only skipped
        if grep -q "passed" "$log_file"; then
            skipped_only=false
        elif grep -q "skipped" "$log_file"; then
            skipped_only=true
        elif grep -q "deselected" "$log_file" && ! grep -q "passed" "$log_file"; then
            skipped_only=true
        fi
    fi
    
    # Write result to file (for collection later)
    if [ "$skipped_only" = true ]; then
        echo "SKIPPED" > "$result_file"
        echo -e "${GREY}○ [GPUs ${gpu_list}] ${model_name}: SKIPPED${NC}"
    elif [ $exit_code -eq 0 ]; then
        echo "SUCCESS" > "$result_file"
        echo -e "${GREEN}✓ [GPUs ${gpu_list}] ${model_name}: SUCCESS${NC}"
    else
        echo "FAILED (exit code: $exit_code)" > "$result_file"
        echo -e "${RED}✗ [GPUs ${gpu_list}] ${model_name}: FAILED (exit code: $exit_code)${NC}"
    fi
}

# Get number of models
NUM_MODELS=${#MODEL_NAMES[@]}

# Track PIDs for waiting
declare -a PIDS=()
declare -a SLOTS=()

# Launch tests in parallel, cycling through available GPU pairs
for i in "${!MODEL_NAMES[@]}"; do
    model_name="${MODEL_NAMES[$i]}"
    test_file="${MODELS[$model_name]}"
    slot_id=$((i % NUM_PARALLEL))
    
    # If we've used all slots, wait for a slot to free up
    if [ ${#PIDS[@]} -ge $NUM_PARALLEL ]; then
        # Wait for any one process to complete
        wait -n 2>/dev/null || wait "${PIDS[0]}"
        # Remove completed PIDs (simplified: just clear and rebuild)
        NEW_PIDS=()
        for pid in "${PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                NEW_PIDS+=("$pid")
            fi
        done
        PIDS=("${NEW_PIDS[@]}")
    fi
    
    run_test "$model_name" "$test_file" "$slot_id" &
    PIDS+=($!)
done

# Wait for all remaining background jobs to complete
echo ""
echo "Waiting for all tests to complete..."
wait

# Print summary
echo ""
echo "=========================================="
echo "  SUMMARY"
echo "=========================================="
echo ""

success_count=0
fail_count=0
skip_count=0

for model_name in "${MODEL_NAMES[@]}"; do
    result_file="$RESULTS_DIR/${model_name}.result"
    if [ -f "$result_file" ]; then
        result=$(cat "$result_file")
        if [[ "$result" == "SUCCESS" ]]; then
            echo -e "${GREEN}✓ ${model_name}: ${result}${NC}"
            ((success_count++))
        elif [[ "$result" == "SKIPPED" ]]; then
            echo -e "${GREY}○ ${model_name}: ${result}${NC}"
            ((skip_count++))
        else
            echo -e "${RED}✗ ${model_name}: ${result}${NC}"
            # Show last few lines of error
            echo -e "${DIM}  Error snippet:"
            tail -n 5 "$RESULTS_DIR/${model_name}.log" | while read -r line; do echo -e "    ${DIM}${line}${NC}"; done
            ((fail_count++))
        fi
    else
        echo -e "${RED}✗ ${model_name}: NO RESULT (test may have crashed)${NC}"
        ((fail_count++))
    fi
done

echo ""
echo "-------------------------------------------"
echo -e "Total: ${GREEN}${success_count} passed${NC}, ${GREY}${skip_count} skipped${NC}, ${RED}${fail_count} failed${NC}"
echo "=========================================="

# Show logs for failed tests (full paths for clickable links)
if [ $fail_count -gt 0 ]; then
    echo ""
    echo "Failed test logs (full paths):"
    for model_name in "${MODEL_NAMES[@]}"; do
        result_file="$RESULTS_DIR/${model_name}.result"
        if [ -f "$result_file" ] && [ "$(cat "$result_file")" != "SUCCESS" ] && [ "$(cat "$result_file")" != "SKIPPED" ]; then
            echo "  $RESULTS_DIR/${model_name}.log"
        fi
    done
fi

# Exit with failure if any tests failed
if [ $fail_count -gt 0 ]; then
    exit 1
fi