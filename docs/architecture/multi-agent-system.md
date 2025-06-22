# Plexe Multi-Agent Architecture

<div align="center">

*A framework for autonomous machine learning using specialized agents*
</div>

## 📚 Table of Contents

- [Overview](#overview)
- [Architecture Diagram](#architecture-diagram)
- [Key Components](#key-components)
  - [EDA Agent](#eda-agent)
  - [Schema Resolver Agent](#schema-resolver-agent)
  - [Feature Engineering Agent](#feature-engineering-agent)
  - [Dataset Splitter Agent](#dataset-splitter-agent)
  - [Manager Agent (Orchestrator)](#manager-agent-orchestrator)
  - [ML Research Scientist Agent](#ml-research-scientist-agent)
  - [ML Engineer Agent](#ml-engineer-agent)
  - [ML Operations Engineer Agent](#ml-operations-engineer-agent)
  - [Model Tester Agent](#model-tester-agent)
  - [Object Registry](#object-registry)
  - [Tool System](#tool-system)
- [Workflow](#workflow)
- [Implementation Details](#implementation-details)
- [Extending the System](#extending-the-system)
- [References](#references)

## Overview

Plexe employs a sophisticated multi-agent architecture to automate the end-to-end machine learning development process. Instead of relying on a single large language model (LLM) to handle all aspects of ML development, Plexe uses a team of specialized agents, each designed for specific tasks in the ML lifecycle.

This approach offers several advantages:
- **Specialization**: Each agent focuses on what it does best
- **Modularity**: Components can be improved or replaced independently
- **Scalability**: The system can handle increasingly complex ML tasks
- **Explainability**: Clear separation of concerns helps trace decisions

## Architecture Diagram

```mermaid
graph TD
    User([User]) --> |"Intent & Datasets"| ModelBuilder["ModelBuilder"]
    User --> |"Intent & Datasets"| Model["Model Class deprecated"]
    
    subgraph "Multi-Agent System"
        ModelBuilder --> |build| Orchestrator["Manager Agent"]
        Model --> |build deprecated| ModelBuilder
        Orchestrator --> |"Schema Task"| SchemaResolver["Schema Resolver"]
        Orchestrator --> |"EDA Task"| EDA["EDA Agent"]
        Orchestrator --> |"Feature Task"| FE["Feature Engineer"]
        Orchestrator --> |"Plan Task"| MLS["ML Researcher"]
        Orchestrator --> |"Split Task"| DS["Dataset Splitter"]
        Orchestrator --> |"Implement Task"| MLE["ML Engineer"]
        Orchestrator --> |"Inference Task"| MLOPS["ML Operations"]
        Orchestrator --> |"Testing Task"| Tester["Model Tester"]
        
        SchemaResolver --> |"Schemas"| Orchestrator
        EDA --> |"Analysis & Reports"| Orchestrator
        FE --> |"Transformed Datasets"| Orchestrator
        MLS --> |"Multiple Solution Plans"| Orchestrator
        DS --> |"Split Datasets"| Orchestrator
        MLE --> |"Multiple Trained Models"| Orchestrator
        MLOPS --> |"Inference Code for Each Model"| Orchestrator
        Tester --> |"Evaluation Reports for All Models"| Orchestrator
    end
    
    subgraph Registry["Object Registry"]
        Datasets[(Datasets)]
        EdaReports[(EDA Reports)]
        Solutions[(Solution Objects)]
        Artifacts[(Model Artifacts)]
        Code[(Code Snippets)]
        Schemas[(I/O Schemas)]
    end
    
    subgraph Tools["Tool System"]
        MetricTool["Metric Selection"]
        CodeGenTool["Code Generation"]
        ValidationTool["Validation"]
        ExecutionTool["Execution"]
        DatasetTool["Dataset Handling"]
        ReviewTool["Model Review"]
        SolutionTool["Solution Management"]
    end
    
    Orchestrator <--> Registry
    Orchestrator <--> Tools
    MLS <--> Tools
    MLS <--> EdaReports
    MLE <--> Tools
    MLE <--> EdaReports
    MLOPS <--> Tools
    SchemaResolver <--> Registry
    SchemaResolver <--> Tools
    SchemaResolver <--> EdaReports
    EDA <--> Registry
    EDA <--> Tools
    DS <--> Registry
    DS <--> Tools
    
    Orchestrator --> |"Best Solution Selected"| Result([Trained Model])
    Result --> Model
    Model --> |predict| Prediction([Predictions])
```

## Key Components

### EDA Agent

**Class**: `EdaAgent` 
**Type**: `CodeAgent`

The EDA Agent performs exploratory data analysis on datasets early in the workflow:

```python
self.eda_agent = EdaAgent(
    model_id=self.orchestrator_model_id,
    verbose=verbose,
    chain_of_thought_callable=chain_of_thought_callable,
).agent
```

**Responsibilities**:
- Analyzing datasets to understand structure, distributions, and relationships
- Identifying data quality issues, outliers, and missing values
- Generating key insights about the data
- Providing recommendations for preprocessing and modeling
- Registering EDA reports in the Object Registry for use by downstream agents

### Schema Resolver Agent

**Class**: `SchemaResolverAgent`
**Type**: `CodeAgent`

The Schema Resolver Agent infers input and output schemas from intent and dataset samples:

```python
self.schema_resolver_agent = SchemaResolverAgent(
    model_id=self.orchestrator_model_id,
    verbose=verbose,
    chain_of_thought_callable=chain_of_thought_callable,
).agent
```

**Responsibilities**:
- Analyzing the problem description and sample data
- Inferring appropriate input and output schemas
- Registering schemas with the Object Registry
- Providing automatic schema resolution when schemas aren't specified

### Feature Engineering Agent

**Class**: `FeatureEngineeringAgent`
**Type**: `CodeAgent`

The Feature Engineering Agent transforms raw datasets into optimized features for model training:

```python
self.feature_engineering_agent = FeatureEngineeringAgent(
    model_id=self.ml_engineer_model_id,
    verbose=verbose,
    chain_of_thought_callable=self.chain_of_thought_callable,
).agent
```

**Responsibilities**:
- Analyzing datasets based on EDA insights
- Creating feature transformations to improve model performance
- Handling data cleaning, encoding, and feature creation
- Preserving data integrity during transformations
- Registering transformed datasets in the Object Registry
- Storing transformation code for inclusion in the final model

### Dataset Splitter Agent

**Class**: `DatasetSplitterAgent`
**Type**: `CodeAgent`

The Dataset Splitter Agent handles the intelligent partitioning of datasets:

```python
self.dataset_splitter_agent = DatasetSplitterAgent(
    model_id=self.orchestrator_model_id,
    verbose=verbose,
    chain_of_thought_callable=self.chain_of_thought_callable,
).agent
```

**Responsibilities**:
- Analyzing datasets to determine appropriate splitting strategies
- Handling specialized splitting needs (time-series, imbalanced data)
- Creating train/validation/test splits with proper stratification
- Registering split datasets in the Object Registry for downstream use

### Manager Agent (Orchestrator)

**Class**: `CodeAgent`  
**Type**: `CodeAgent`

The Manager Agent serves as the central coordinator for the entire ML development process:

```python
self.manager_agent = CodeAgent(
    name="Orchestrator",
    model=LiteLLMModel(model_id=self.orchestrator_model_id),
    tools=[
        get_select_target_metric(self.tool_model_id),
        get_review_finalised_model(self.tool_model_id),
        get_latest_datasets,
        get_solution_performances,
        register_best_solution,
        format_final_orchestrator_agent_response,
    ],
    managed_agents=[
        self.eda_agent,
        self.schema_resolver_agent,
        self.feature_engineering_agent,
        self.ml_research_agent,
        self.dataset_splitter_agent,
        self.mle_agent,
        self.mlops_engineer,
        self.model_tester_agent,
    ],
    add_base_tools=False,
    verbosity_level=self.orchestrator_verbosity,
    additional_authorized_imports=config.code_generation.authorized_agent_imports,
    max_steps=self.max_steps,
    planning_interval=7,
    step_callbacks=[self.chain_of_thought_callable],
)
```

**Responsibilities**:
- Initializing the problem based on user intent
- Selecting appropriate metrics
- Coordinating specialist agents including schema resolver and EDA agents
- Making decisions about which solution approach to pursue
- Collecting and integrating the final model artifacts

### ML Research Scientist Agent

**Class**: `ModelPlannerAgent`  
**Type**: `ToolCallingAgent`

This agent specializes in solution planning and strategy:

```python
self.ml_research_agent = ModelPlannerAgent(
    model_id=ml_researcher_model_id,
    verbose=verbose,
    chain_of_thought_callable=chain_of_thought_callable,
).agent
```

**Responsibilities**:
- Analyzing the problem and available datasets
- Brainstorming possible ML approaches
- Developing detailed solution plans
- Providing rationales for suggested approaches

### ML Engineer Agent

**Class**: `ModelTrainerAgent`  
**Type**: `CodeAgent`

This agent handles the implementation and training of models:

```python
self.mle_agent = ModelTrainerAgent(
    ml_engineer_model_id=self.ml_engineer_model_id,
    tool_model_id=self.tool_model_id,
    distributed=self.distributed,
    verbose=verbose,
    chain_of_thought_callable=chain_of_thought_callable,
    schema_resolver_agent=self.schema_resolver_agent,
).agent
```

**Responsibilities**:
- Generating training code based on solution plans
- Training models and evaluating performance
- Handling data preprocessing
- Troubleshooting and fixing issues
- Creating model artifacts

### ML Operations Engineer Agent

**Class**: `ModelPackagerAgent`  
**Type**: `CodeAgent`

This agent focuses on productionizing the model through inference code:

```python
self.mlops_engineer = ModelPackagerAgent(
    model_id=self.ml_ops_engineer_model_id,
    tool_model_id=self.tool_model_id,
    verbose=verbose,
    chain_of_thought_callable=chain_of_thought_callable,
    schema_resolver_agent=self.schema_resolver_agent,
).agent
```

**Responsibilities**:
- Generating inference code for trained models
- Implementing model loading logic
- Creating input preprocessing and output postprocessing
- Validating inference code correctness

### Model Tester Agent

**Class**: `ModelTesterAgent`  
**Type**: `CodeAgent`

This agent focuses on comprehensive testing and evaluation of finalized ML models:

```python
self.model_tester_agent = ModelTesterAgent(
    model_id=self.ml_engineer_model_id,
    verbose=verbose,
    chain_of_thought_callable=self.chain_of_thought_callable,
).agent
```

**Responsibilities**:
- Evaluating model performance on test datasets
- Performing quality analysis and robustness testing
- Generating comprehensive evaluation reports
- Testing model behavior with edge cases
- Providing insights about model limitations and strengths

### Object Registry

**Class**: `ObjectRegistry`

The Object Registry provides a shared repository for storing and retrieving objects across the multi-agent system:

```python
class ObjectRegistry:
    """
    Registry for storing and retrieving objects by name.

    This class implements the Singleton pattern so that registry instances are shared
    across the application. It provides methods for registering, retrieving, and
    managing objects in a type-safe manner.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ObjectRegistry, cls).__new__(cls)
            cls._instance._items = {}
        return cls._instance
```

**Key Features**:
- Type-safe storage and retrieval with URI-based namespacing
- Singleton pattern ensures shared access across agents
- Support for immutable objects to prevent modification
- Registration of multiple item types (datasets, artifacts, code, schemas, EDA reports)
- Batch operations with register_multiple and get_multiple
- Deep copy support for immutable items to prevent accidental modifications

### Tool System

The system includes specialized tools that agents can use to perform specific tasks, implemented using factory patterns:

**Metric Selection Tool**:
```python
def get_select_target_metric(model_id: str) -> Callable:
    """Factory function that returns a tool for selecting appropriate target metrics."""
    @tool
    def select_target_metric(task: str) -> Dict:
        """Selects the appropriate target metric to optimise for the given task."""
```

**Code Generation Tools**:
```python
def get_training_code_generation_tool(llm_to_use: str) -> Callable:
    """Factory function that returns a tool for generating training code."""
    @tool
    def generate_training_code(
        task: str, solution_plan: str, train_datasets: List[str], 
        validation_datasets: List[str]
    ) -> str:
        """Generates training code based on the solution plan."""
```

**Dataset Tools**:
```python
@tool
def register_split_datasets(
    dataset_names: List[str],
    train_datasets: List[pd.DataFrame],
    validation_datasets: List[pd.DataFrame],
    test_datasets: List[pd.DataFrame],
) -> Dict[str, List[str]]:
    """Register train, validation, and test datasets in the object registry."""
```

**Execution Tools**:
```python
def get_executor_tool(distributed: bool) -> Callable:
    """Factory function that returns the appropriate executor tool."""
    if distributed:
        return execute_training_code_distributed
    return execute_training_code
```

## Workflow

The multi-agent workflow follows these key steps:

1. **Initialization**:
   - User creates a `ModelBuilder` instance or `Model` instance with intent and datasets
   - User calls `ModelBuilder.build()` or `model.build()` (deprecated) to start the process

2. **Orchestration**:
   - `ModelBuilder` (preferred) or `Model.build()` (deprecated) initializes the process
   - Manager Agent coordinates the entire process and tasks specialist agents based on workflow requirements

3. **Schema Resolution**:
   - If schemas aren't provided, SchemaResolverAgent infers them
   - The agent analyzes the problem description and sample data
   - Schemas are registered in the Object Registry

4. **Exploratory Data Analysis**:
   - EdaAgent analyzes datasets to understand structure and characteristics
   - Generates insights about data patterns, quality issues, and modeling considerations
   - EDA reports are registered in the Object Registry for use by other agents

5. **Feature Engineering** (Optional):
   - Feature Engineering Agent transforms raw datasets based on EDA insights
   - Creates new features, handles encoding, and cleans data
   - Registers transformed datasets in the Object Registry
   - Stores transformation code for inclusion in the final model

6. **Dataset Splitting**:
   - Dataset Splitter Agent analyzes data characteristics
   - Creates appropriate train/validation/test splits
   - Registers split datasets in the Object Registry

7. **Solution Planning**:
   - ML Research Scientist analyzes the problem and proposes multiple solution approaches
   - Manager Agent evaluates and selects the best approaches based on requirements
   - Solution plans are registered for implementation

8. **Model Implementation**:
   - ML Engineer generates and executes training code for each solution plan
   - Multiple models may be trained to compare performance
   - Model artifacts and performance metrics are captured for each solution

9. **Inference Code Generation**:
   - ML Operations Engineer generates compatible inference code for each trained solution
   - Code is validated with sample inputs for correctness
   - Predictor instances are created and tested

10. **Model Testing**:
   - Model Tester Agent evaluates all finalized models on test data
   - Comprehensive evaluation reports are generated for performance comparison
   - Quality analysis and robustness testing are performed

11. **Solution Selection and Finalization**:
   - Manager Agent compares all solution performances and selects the best performing model
   - The best solution is registered as the final model
   - All artifacts, code, and evaluation results are collected
   - Model metadata is extracted and stored
   - Completed model is returned to the user

## Implementation Details

### Agent Communication

The system uses a hierarchical communication pattern:

```
User → Model → Manager Agent → Specialist Agents → Manager Agent → Model → User
```

Each agent communicates through structured task descriptions and responses:

```python
result = self.manager_agent.run(
    task=agent_prompt,
    additional_args={
        "intent": self.intent,
        "working_dir": self.working_dir,
        "input_schema": format_schema(self.input_schema),
        "output_schema": format_schema(self.output_schema),
        "max_iterations": max_iterations,
        "timeout": timeout,
        "run_timeout": run_timeout,
    },
)
```

### Code Execution

The system executes generated code in isolated environments:

```python
class ProcessExecutor(Executor):
    """Execute Python code snippets in an isolated process."""

    def run(self) -> ExecutionResult:
        """Execute code in a subprocess and return results."""
        process = subprocess.Popen(
            [sys.executable, str(self.code_file)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(self.working_dir),
            text=True,
        )
```

For distributed execution, a `RayExecutor` is available:

```python
class RayExecutor(Executor):
    """Execute Python code snippets on a Ray cluster."""
    
    @ray.remote
    def _run_code(code: str, working_dir: str, dataset_files: List[str], timeout: int) -> dict:
        """Ray remote function that executes the code."""
```

### Provider System

The `ProviderConfig` allows different LLMs for different agents:

```python
class ProviderConfig:
    """
    Configuration class for specifying different LLM providers for various agent roles.
    """

    def __init__(
        self,
        default_provider: str = "openai/gpt-4o-mini",
        orchestrator_provider: Optional[str] = None,
        research_provider: Optional[str] = None,
        engineer_provider: Optional[str] = None,
        ops_provider: Optional[str] = None,
        tool_provider: Optional[str] = None,
    ):
        # Default provider is used when specific ones aren't set
        self.default_provider = default_provider

        # Agent-specific providers
        self.orchestrator_provider = orchestrator_provider or default_provider
        self.research_provider = research_provider or default_provider
        self.engineer_provider = engineer_provider or default_provider
        self.ops_provider = ops_provider or default_provider

        # Provider for tool operations
        self.tool_provider = tool_provider or default_provider
```

## Extending the System

The multi-agent architecture in Plexe is designed to be extendable in several ways:

### Adding New Agents

You can create new specialized agents by extending the `PlexeAgent` class:

```python
class CustomPlexeAgent(PlexeAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Add a new specialized agent
        self.custom_agent = ToolCallingAgent(
            name="CustomAgent",
            description="Description of what this agent does",
            model=LiteLLMModel(model_id=self.custom_agent_model_id),
            tools=[custom_tool1, custom_tool2],
            add_base_tools=False,
            verbosity_level=self.specialist_verbosity,
            prompt_templates=get_prompt_templates("toolcalling_agent.yaml", "custom_prompt_templates.yaml"),
        )
        
        # Update manager agent to include new agent
        self.manager_agent.managed_agents.append(self.custom_agent)
```

### Implementing Custom Tools

You can add new tools using the factory pattern with the `@tool` decorator:

```python
def get_custom_tool(model_id: str) -> Callable:
    """Factory function that returns a custom tool."""
    @tool
    def custom_tool(param1: str, param2: int) -> Dict:
        """Description of what this tool does."""
        # Tool implementation
        return {"result": "Output of the tool"}
    return custom_tool
```

### Supporting New Model Types

To support new model types, extend the execution and validation systems:

```python
class CustomModelValidator(Validator):
    """A validator for custom model types."""
    
    def validate(self, code: str, **kwargs) -> ValidationResult:
        # Validation logic for custom models
        pass
```

## References

- [PlexeAgent Class Definition](plexe/agents/agents.py)
- [Model Class Definition](plexe/models.py)
- [ModelBuilder Class Definition](plexe/model_builder.py)
- [EdaAgent Definition](plexe/agents/dataset_analyser.py)
- [SchemaResolverAgent Definition](plexe/agents/schema_resolver.py)
- [FeatureEngineeringAgent Definition](plexe/agents/feature_engineer.py)
- [DatasetSplitterAgent Definition](plexe/agents/dataset_splitter.py)
- [ModelTrainerAgent Definition](plexe/agents/model_trainer.py)
- [ModelPackagerAgent Definition](plexe/agents/model_packager.py)
- [ModelPlannerAgent Definition](plexe/agents/model_planner.py)
- [ModelTesterAgent Definition](plexe/agents/model_tester.py)
- [Tool Definitions](plexe/tools/)
- [Dataset Tools](plexe/tools/datasets.py)
- [Validation Tools](plexe/tools/validation.py)
- [Testing Tools](plexe/tools/testing.py)
- [Executor Implementation](plexe/internal/models/execution/)
- [Object Registry](plexe/core/object_registry.py)
