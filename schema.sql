--
-- PostgreSQL database dump
--

-- Dumped from database version 17.5
-- Dumped by pg_dump version 17.5

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: alertness_data_for_visualization; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.alertness_data_for_visualization (
    id integer NOT NULL,
    user_id uuid NOT NULL,
    "timestamp" timestamp with time zone NOT NULL,
    awake boolean NOT NULL,
    "g_PD" real NOT NULL,
    "P0_values" real NOT NULL,
    "P_t_caffeine" real NOT NULL,
    "P_t_no_caffeine" real NOT NULL,
    "P_t_real" real NOT NULL
);


ALTER TABLE public.alertness_data_for_visualization OWNER TO postgres;

--
-- Name: alertness_data_for_visualization_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.alertness_data_for_visualization_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.alertness_data_for_visualization_id_seq OWNER TO postgres;

--
-- Name: alertness_data_for_visualization_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.alertness_data_for_visualization_id_seq OWNED BY public.alertness_data_for_visualization.id;

--
-- Insert sample data into alertness_data_for_visualization
--

INSERT INTO public.alertness_data_for_visualization (user_id, "timestamp", awake, "g_PD", "P0_values", "P_t_caffeine", "P_t_no_caffeine", "P_t_real")
VALUES 
    ('550e8400-e29b-41d4-a716-446655440000', '2025-03-30 14:00:00+08', TRUE, 0.5, 1.0, 0.2, 0.3, 0.4),
    ('550e8400-e29b-41d4-a716-446655440000', '2025-03-31 14:00:00+08', FALSE, 0.6, 1.1, 0.3, 0.4, 0.5);


--
-- Name: recommendations_caffeine; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.recommendations_caffeine (
    id integer NOT NULL,
    user_id uuid NOT NULL,
    recommended_caffeine_amount integer NOT NULL,
    recommended_caffeine_intake_timing time with time zone NOT NULL
);


ALTER TABLE public.recommendations_caffeine OWNER TO postgres;

--
-- Name: recommendations_caffeine_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.recommendations_caffeine_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.recommendations_caffeine_id_seq OWNER TO postgres;

--
-- Name: recommendations_caffeine_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.recommendations_caffeine_id_seq OWNED BY public.recommendations_caffeine.id;


--
-- Name: users_real_sleep_data; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.users_real_sleep_data (
    id integer NOT NULL,
    user_id uuid NOT NULL,
    start_time timestamp with time zone NOT NULL,
    end_time timestamp with time zone NOT NULL
);


ALTER TABLE public.users_real_sleep_data OWNER TO postgres;

--
-- Name: user_real_sleep_data_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.user_real_sleep_data_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.user_real_sleep_data_id_seq OWNER TO postgres;

--
-- Name: user_real_sleep_data_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.user_real_sleep_data_id_seq OWNED BY public.users_real_sleep_data.id;


--
-- Name: users; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.users (
    id integer NOT NULL,
    user_id uuid NOT NULL,
    name text NOT NULL,
    email text NOT NULL,
    age integer NOT NULL,
    weight numeric NOT NULL,
    created_at timestamp with time zone NOT NULL
);


ALTER TABLE public.users OWNER TO postgres;

--
-- Name: users_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.users_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.users_id_seq OWNER TO postgres;

--
-- Name: users_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.users_id_seq OWNED BY public.users.id;


--
-- Name: users_pvt_results; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.users_pvt_results (
    id integer NOT NULL,
    user_id uuid NOT NULL,
    mean_rt real NOT NULL,
    lapses integer NOT NULL,
    false_starts integer NOT NULL,
    test_at timestamp with time zone NOT NULL,
    device text NOT NULL
);


ALTER TABLE public.users_pvt_results OWNER TO postgres;

--
-- Name: users_pvt_results_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.users_pvt_results_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.users_pvt_results_id_seq OWNER TO postgres;

--
-- Name: users_pvt_results_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.users_pvt_results_id_seq OWNED BY public.users_pvt_results.id;


--
-- Name: users_real_time_intake; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.users_real_time_intake (
    id integer NOT NULL,
    user_id uuid NOT NULL,
    drink_name text NOT NULL,
    caffeine_amount integer NOT NULL,
    taking_timestamp timestamp with time zone NOT NULL
);


ALTER TABLE public.users_real_time_intake OWNER TO postgres;

--
-- Name: users_real_time_intake_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.users_real_time_intake_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.users_real_time_intake_id_seq OWNER TO postgres;

--
-- Name: users_real_time_intake_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.users_real_time_intake_id_seq OWNED BY public.users_real_time_intake.id;


--
-- Name: users_target_waking_period; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.users_target_waking_period (
    id integer NOT NULL,
    user_id uuid,
    target_start_time time with time zone,
    target_end_time time with time zone
);


ALTER TABLE public.users_target_waking_period OWNER TO postgres;

--
-- Name: alertness_data_for_visualization id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.alertness_data_for_visualization ALTER COLUMN id SET DEFAULT nextval('public.alertness_data_for_visualization_id_seq'::regclass);


--
-- Name: recommendations_caffeine id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.recommendations_caffeine ALTER COLUMN id SET DEFAULT nextval('public.recommendations_caffeine_id_seq'::regclass);


--
-- Name: users id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.users ALTER COLUMN id SET DEFAULT nextval('public.users_id_seq'::regclass);


--
-- Name: users_pvt_results id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.users_pvt_results ALTER COLUMN id SET DEFAULT nextval('public.users_pvt_results_id_seq'::regclass);


--
-- Name: users_real_sleep_data id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.users_real_sleep_data ALTER COLUMN id SET DEFAULT nextval('public.user_real_sleep_data_id_seq'::regclass);


--
-- Name: users_real_time_intake id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.users_real_time_intake ALTER COLUMN id SET DEFAULT nextval('public.users_real_time_intake_id_seq'::regclass);


--
-- Name: alertness_data_for_visualization alertness_data_for_visualization_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.alertness_data_for_visualization
    ADD CONSTRAINT alertness_data_for_visualization_pkey PRIMARY KEY (id);


--
-- Name: recommendations_caffeine recommendations_caffeine_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.recommendations_caffeine
    ADD CONSTRAINT recommendations_caffeine_pkey PRIMARY KEY (id);


--
-- Name: users_real_sleep_data user_real_sleep_data_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.users_real_sleep_data
    ADD CONSTRAINT user_real_sleep_data_pkey PRIMARY KEY (id);


--
-- Name: users users_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_pkey PRIMARY KEY (id);


--
-- Name: users_pvt_results users_pvt_results_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.users_pvt_results
    ADD CONSTRAINT users_pvt_results_pkey PRIMARY KEY (id);


--
-- Name: users_real_time_intake users_real_time_intake_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.users_real_time_intake
    ADD CONSTRAINT users_real_time_intake_pkey PRIMARY KEY (id);


--
-- Name: users_target_waking_period users_target_waking_period_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.users_target_waking_period
    ADD CONSTRAINT users_target_waking_period_pkey PRIMARY KEY (id);


--
-- PostgreSQL database dump complete
--

